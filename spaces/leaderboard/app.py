"""
ColliderML Benchmark Leaderboard — Gradio HuggingFace Space.

Browse task leaderboards, submit predictions, reproduce others' results.

All scoring happens on the backend — this Space is just a frontend that
forwards the uploaded file and the user's HF OAuth token.

The task list is loaded from ``colliderml.tasks`` at startup so adding a
new task in the library is enough to surface it in the UI (no Space
redeploy needed beyond a pin-bump on the ``colliderml`` package).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import gradio as gr
import pandas as pd
import requests

BACKEND = os.environ.get("COLLIDERML_BACKEND", "https://api.colliderml.com").rstrip("/")


def _discover_tasks() -> List[str]:
    """Load the task list from the installed colliderml package.

    Falls back to a hardcoded default so the Space still boots if the
    ``colliderml`` import fails for any reason (partial install, network
    blip during first boot, etc.).
    """
    try:
        import colliderml.tasks as _tasks
    except Exception:
        return [
            "tracking",
            "jets",
            "anomaly",
            "tracking_latency",
            "tracking_small",
            "data_loading",
        ]
    return list(_tasks.list_tasks())


TASKS: List[str] = _discover_tasks()

TASK_DESCRIPTIONS = {
    "tracking": "Track reconstruction on ttbar pu200 — TrackML weighted efficiency, fake/dup rates",
    "jets": "Jet flavour classification on ttbar pu0 — b-tag AUC, rejection rates",
    "anomaly": "BSM anomaly detection — AUROC, signal efficiency @ 1% FPR",
    "tracking_latency": "Wall-clock time on 1000 eval events",
    "tracking_small": "Best efficiency under a parameter budget",
    "data_loading": "Throughput of data loading pipelines",
}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def fetch_leaderboard(task: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{BACKEND}/v1/leaderboard/{task}?limit=100", timeout=20)
        r.raise_for_status()
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    rows = r.json()
    if not rows:
        return pd.DataFrame({"info": ["No submissions yet. Be the first!"]})

    flat = []
    for row in rows:
        scores = row.get("scores") or {}
        if isinstance(scores, str):
            try:
                scores = json.loads(scores)
            except json.JSONDecodeError:
                scores = {}
        flat.append({
            "submitter": row["hf_username"],
            **scores,
            "credits_earned": row.get("credits_earned", 0),
            "baseline": row.get("is_baseline", False),
            "submitted_at": row["submitted_at"],
            "submission_id": row["id"],
        })
    return pd.DataFrame(flat)


def submit(task: str, file, oauth_token: Optional[gr.OAuthToken]):
    if oauth_token is None:
        return "Please sign in with HuggingFace first."
    if file is None:
        return "No file uploaded."
    try:
        with open(file.name, "rb") as f:
            payload = f.read()
        r = requests.post(
            f"{BACKEND}/v1/benchmark/{task}/submit",
            files={"predictions": ("predictions.parquet", payload)},
            data={"local_scores": "{}"},
            headers=_headers(oauth_token.token),
            timeout=300,
        )
    except Exception as e:
        return f"Upload error: {e}"
    if r.status_code >= 400:
        return f"Backend error {r.status_code}: {r.text}"
    data = r.json()
    msg = (
        f"**Submission accepted!**  \n"
        f"- ID: `{data['submission_id']}`  \n"
        f"- Scores: {data['scores']}  \n"
        f"- Credits earned: **{data.get('credits_earned', 0)}**\n"
    )
    if data.get("deduplicated"):
        msg += "- *Same file as a previous submission — no additional credits.*\n"
    return msg


def reproduce(task: str, submission_id: str, file, oauth_token: Optional[gr.OAuthToken]):
    if oauth_token is None:
        return "Please sign in with HuggingFace first."
    if not submission_id:
        return "Enter the submission ID to reproduce."
    if file is None:
        return "No file uploaded."
    try:
        with open(file.name, "rb") as f:
            payload = f.read()
        r = requests.post(
            f"{BACKEND}/v1/benchmark/{task}/reproduce/{submission_id}",
            files={"predictions": ("predictions.parquet", payload)},
            headers=_headers(oauth_token.token),
            timeout=300,
        )
    except Exception as e:
        return f"Upload error: {e}"
    if r.status_code >= 400:
        return f"Backend error {r.status_code}: {r.text}"
    data = r.json()
    msg = (
        f"**Reproduction complete**  \n"
        f"- Within 2% tolerance: **{data['within_tolerance']}**  \n"
        f"- Original scores: {data['original_scores']}  \n"
        f"- Your scores: {data['reproduced_scores']}  \n"
        f"- Credits earned: **{data.get('credits_earned', 0)}**\n"
    )
    return msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="ColliderML Leaderboard",
    theme=gr.themes.Soft(primary_hue="yellow"),
) as demo:
    gr.Markdown(
        """
        # 🏆 ColliderML Benchmark Leaderboard

        Beat any metric → earn credits. Reproduce someone's result → earn credits.
        All scoring is server-side against held-out truth.
        """
    )

    gr.LoginButton()

    for task in TASKS:
        with gr.Tab(task):
            gr.Markdown(
                f"**{task}** — {TASK_DESCRIPTIONS.get(task, 'See the library docs for details.')}"
            )

            refresh_btn = gr.Button("Refresh leaderboard")
            board = gr.DataFrame(
                value=fetch_leaderboard(task),
                label=f"Top submissions — {task}",
                interactive=False,
            )
            refresh_btn.click(
                fn=lambda t=task: fetch_leaderboard(t),
                outputs=board,
            )

            with gr.Accordion("Submit predictions", open=False):
                upload = gr.File(label="predictions.parquet", file_types=[".parquet"])
                submit_btn = gr.Button("Submit", variant="primary")
                submit_out = gr.Markdown()
                submit_btn.click(
                    fn=lambda f, t=task: submit(t, f, None),
                    inputs=[upload],
                    outputs=submit_out,
                )

            with gr.Accordion("Reproduce a submission", open=False):
                submission_id = gr.Textbox(label="Submission ID to reproduce")
                repro_file = gr.File(label="Your predictions.parquet", file_types=[".parquet"])
                repro_btn = gr.Button("Reproduce")
                repro_out = gr.Markdown()
                repro_btn.click(
                    fn=lambda sid, f, t=task: reproduce(t, sid, f, None),
                    inputs=[submission_id, repro_file],
                    outputs=repro_out,
                )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
