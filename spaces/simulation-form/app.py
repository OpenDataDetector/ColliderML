"""
ColliderML Simulation Service — Gradio HuggingFace Space.

Two tabs:
    1. Simulation form — pick channel/events/pileup, submit, track status.
    2. Chat agent — natural-language interface via Anthropic Claude with
       tool use (calls the same backend under the hood).

Authentication is HuggingFace OAuth. The OAuth token is forwarded as a
bearer token to the backend, which verifies it and does all the real work.
"""

from __future__ import annotations

import os
from typing import Optional

import gradio as gr
import requests

BACKEND_URL = os.environ.get("COLLIDERML_BACKEND", "https://api.colliderml.com").rstrip("/")

CHANNELS = [
    "higgs_portal",
    "ttbar",
    "zmumu",
    "zee",
    "diphoton",
    "jets",
    "susy_gmsb",
    "hidden_valley",
    "zprime",
    "single_muon",
]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def fetch_me(token: str) -> dict:
    r = requests.get(f"{BACKEND_URL}/v1/me", headers=_headers(token), timeout=20)
    r.raise_for_status()
    return r.json()


def submit_simulation(
    token: str,
    channel: str,
    events: int,
    pileup: int,
    seed: int,
) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/v1/simulate",
        json={"channel": channel, "events": events, "pileup": pileup, "seed": seed},
        headers=_headers(token),
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Backend error {r.status_code}: {r.text}")
    return r.json()


def fetch_request(token: str, request_id: str) -> dict:
    r = requests.get(
        f"{BACKEND_URL}/v1/requests/{request_id}",
        headers=_headers(token),
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Simulation tab handlers
# ---------------------------------------------------------------------------
def login_display(oauth_token: Optional[gr.OAuthToken]):
    if oauth_token is None:
        return "Not signed in. Click **Sign in with HuggingFace** above."
    try:
        me = fetch_me(oauth_token.token)
    except Exception as e:
        return f"Error fetching profile: {e}"
    return (
        f"**Signed in as `{me['hf_username']}`**  \n"
        f"Credits: **{me['credits']:.2f}**"
    )


def on_submit(
    channel: str,
    events: int,
    pileup: int,
    seed: int,
    oauth_token: Optional[gr.OAuthToken],
):
    if oauth_token is None:
        return "Please sign in with HuggingFace first.", None
    try:
        result = submit_simulation(oauth_token.token, channel, events, pileup, seed)
    except Exception as e:
        return f"Error: {e}", None

    message = (
        f"**Submitted!**  \n"
        f"- Request ID: `{result['request_id']}`  \n"
        f"- State: `{result['state']}`  \n"
        f"- Est. credits: **{result['credits_charged']:.2f}**  \n"
        f"- Est. completion: ~{result['estimated_completion_seconds'] // 60} min  \n"
    )
    if result.get("cached"):
        message += "- *This request was deduplicated against a cached result.*\n"
    if result.get("output_hf_repo"):
        message += f"- Output: https://huggingface.co/datasets/{result['output_hf_repo']}\n"
    return message, result["request_id"]


def on_poll(request_id: str, oauth_token: Optional[gr.OAuthToken]):
    if not request_id:
        return "No request ID. Submit a job first."
    if oauth_token is None:
        return "Please sign in with HuggingFace first."
    try:
        data = fetch_request(oauth_token.token, request_id)
    except Exception as e:
        return f"Error: {e}"
    out = (
        f"**Request `{data['id']}`**  \n"
        f"- State: `{data['state']}`  \n"
        f"- Channel: {data['channel']}  \n"
        f"- Events: {data['events']} (pileup={data['pileup']})  \n"
    )
    if data.get("output_hf_repo"):
        out += f"- Output: https://huggingface.co/datasets/{data['output_hf_repo']}\n"
    if data.get("error_message"):
        out += f"- Error: {data['error_message']}\n"
    return out


# ---------------------------------------------------------------------------
# Chat agent
# ---------------------------------------------------------------------------
CHAT_TOOLS = [
    {
        "name": "estimate_compute",
        "description": "Estimate the node-hours (credits) needed for a simulation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": CHANNELS},
                "events": {"type": "integer", "minimum": 1, "maximum": 100000},
                "pileup": {"type": "integer", "minimum": 0, "maximum": 200},
            },
            "required": ["channel", "events"],
        },
    },
    {
        "name": "submit_simulation",
        "description": (
            "Actually submit a simulation request to NERSC. Deducts credits from "
            "the user's balance. Only call after the user has confirmed the parameters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": CHANNELS},
                "events": {"type": "integer", "minimum": 1, "maximum": 100000},
                "pileup": {"type": "integer", "minimum": 0, "maximum": 200},
                "seed": {"type": "integer", "default": 42},
            },
            "required": ["channel", "events"],
        },
    },
    {
        "name": "check_balance",
        "description": "Check the user's current credit balance.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _tool_call(name: str, arguments: dict, oauth_token) -> dict:
    """Execute a tool call against the backend, return a result dict."""
    if oauth_token is None:
        return {"error": "Not signed in"}
    try:
        if name == "check_balance":
            return fetch_me(oauth_token.token)
        if name == "estimate_compute":
            # Dry-run cost estimate that mirrors the backend's cap module.
            from math import ceil
            base = {
                "higgs_portal": 60.0, "ttbar": 90.0, "zmumu": 30.0,
                "zee": 30.0, "diphoton": 30.0, "jets": 45.0,
                "susy_gmsb": 60.0, "hidden_valley": 60.0,
                "zprime": 60.0, "single_muon": 5.0,
            }.get(arguments["channel"], 60.0)
            overhead = 300.0 if arguments["channel"] in (
                "ttbar", "susy_gmsb", "hidden_valley", "zprime"
            ) else 0.0
            pu = arguments.get("pileup", 0)
            seconds = overhead + base * arguments["events"] * (1 + pu / 50)
            credits = round(seconds / 3600, 2)
            return {
                "channel": arguments["channel"],
                "events": arguments["events"],
                "pileup": pu,
                "estimated_credits": credits,
                "estimated_minutes": ceil(seconds / 60),
            }
        if name == "submit_simulation":
            return submit_simulation(
                oauth_token.token,
                arguments["channel"],
                arguments["events"],
                arguments.get("pileup", 0),
                arguments.get("seed", 42),
            )
    except Exception as e:
        return {"error": str(e)}
    return {"error": f"unknown tool {name}"}


def chat_respond(history, message, oauth_token: "gr.OAuthToken | None" = None):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "Chat agent is not configured on this Space (ANTHROPIC_API_KEY not set).",
        })
        return history, ""

    try:
        import anthropic
    except ImportError:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "anthropic package not installed on this Space.",
        })
        return history, ""

    history = history + [{"role": "user", "content": message}]

    client = anthropic.Anthropic()
    system = (
        "You are a helpful assistant for researchers using ColliderML. "
        "You can estimate compute costs, check the user's credit balance, "
        "and submit simulation requests on their behalf. Always confirm "
        "cost and parameters before calling submit_simulation. "
        "1 credit ≈ 100 pu0 events or 20 pu200 events. Users start with 10 credits."
    )

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history if m["role"] in ("user", "assistant")
    ]

    # Tool-use loop
    for _ in range(5):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=CHAT_TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            history.append({"role": "assistant", "content": text})
            return history, ""

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _tool_call(block.name, block.input, oauth_token)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    history.append({
        "role": "assistant",
        "content": "(hit tool-use iteration limit)",
    })
    return history, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="ColliderML Simulation Service",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:
    gr.Markdown(
        """
        # ColliderML Simulation Service

        Submit custom particle physics simulations to NERSC Perlmutter,
        without installing anything locally.
        """
    )

    login_btn = gr.LoginButton()
    status_md = gr.Markdown("Not signed in.")

    demo.load(fn=login_display, inputs=None, outputs=status_md)

    with gr.Tab("Simulate"):
        with gr.Row():
            with gr.Column(scale=2):
                channel = gr.Dropdown(CHANNELS, value="higgs_portal", label="Physics channel")
                events = gr.Number(value=10, minimum=1, maximum=100_000, label="Events", precision=0)
                pileup = gr.Slider(0, 200, value=0, step=10, label="Pileup")
                seed = gr.Number(value=42, precision=0, label="Seed")
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=3):
                submit_output = gr.Markdown()
                last_request_id = gr.Textbox(label="Last request ID", interactive=False)
                poll_btn = gr.Button("Refresh status")
                poll_output = gr.Markdown()

        submit_btn.click(
            fn=on_submit,
            inputs=[channel, events, pileup, seed],
            outputs=[submit_output, last_request_id],
        )
        poll_btn.click(
            fn=on_poll,
            inputs=[last_request_id],
            outputs=poll_output,
        )

    with gr.Tab("Chat"):
        gr.Markdown(
            """
            Describe what you need in plain English — the agent will estimate
            compute, check your balance, and submit the request after you confirm.

            Example: *"I need 1000 ttbar events with pileup 200 for jet tagging."*
            """
        )
        chatbot = gr.Chatbot(type="messages", height=450)
        msg = gr.Textbox(label="Message", placeholder="Ask me to simulate something...")
        clear = gr.Button("Clear")

        # Gradio auto-injects gr.OAuthToken into any annotated default arg
        # at runtime; do NOT list it in the explicit inputs.
        msg.submit(chat_respond, [chatbot, msg], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
