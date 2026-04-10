"""
ColliderML Model Zoo — Gradio HF Space.

Thin browser over ``huggingface_hub.list_models(filter="colliderml")``.
Sortable, filterable, with direct links to each model card.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd
from huggingface_hub import list_models


def fetch_models(task_filter: str = "any") -> pd.DataFrame:
    try:
        models = list(list_models(filter="colliderml", limit=500, full=True))
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

    rows = []
    for m in models:
        tags = m.tags or []
        task = next((t for t in tags if t in ("tracking", "jets", "anomaly")), "general")
        if task_filter != "any" and task != task_filter:
            continue
        rows.append({
            "model": m.modelId,
            "task": task,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
            "pipeline": getattr(m, "pipeline_tag", "") or "",
            "url": f"https://huggingface.co/{m.modelId}",
        })

    if not rows:
        return pd.DataFrame(
            {"info": ["No models tagged `colliderml` yet. Add the tag to your model card!"]}
        )

    df = pd.DataFrame(rows)
    return df.sort_values("downloads", ascending=False).reset_index(drop=True)


with gr.Blocks(
    title="ColliderML Model Zoo",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:
    gr.Markdown(
        """
        # 🦒 ColliderML Model Zoo

        Models tagged `colliderml` on HuggingFace. Tag your own to appear here:

        ```yaml
        ---
        tags:
          - colliderml
        ---
        ```
        """
    )
    with gr.Row():
        task_filter = gr.Dropdown(
            ["any", "tracking", "jets", "anomaly", "general"],
            value="any",
            label="Filter by task",
        )
        refresh_btn = gr.Button("Refresh")
    table = gr.DataFrame(value=fetch_models())

    refresh_btn.click(fetch_models, inputs=task_filter, outputs=table)
    task_filter.change(fetch_models, inputs=task_filter, outputs=table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
