"""
ColliderML Model Zoo — Gradio HF Space.

Thin browser over ``huggingface_hub.list_models(filter="colliderml")``.
Sortable, filterable, with direct links to each model card.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, list_models

logger = logging.getLogger(__name__)

HF_RESULTS_DATASET = os.environ.get("COLLIDERML_RESULTS_DATASET", "CERN/colliderml-benchmark-results")


def _load_best_scores() -> dict[str, dict]:
    """Read the results dataset and return {username: {task: primary_score}}."""
    try:
        api = HfApi()
        tree = api.list_repo_tree(HF_RESULTS_DATASET, repo_type="dataset", recursive=True)
        json_files = [f for f in tree if f.rfilename.endswith(".json")]
        best: dict[str, dict] = {}
        for f in json_files:
            path = hf_hub_download(HF_RESULTS_DATASET, f.rfilename, repo_type="dataset")
            r = json.loads(Path(path).read_text())
            model = r.get("model_repo_id") or ""
            if not model:
                continue
            task = r.get("task", "?")
            scores = r.get("scores", {})
            primary = list(scores.values())[0] if scores else 0
            key = model
            if key not in best or primary > list(best[key].get("scores", {}).values())[0]:
                best[key] = {"task": task, "scores": scores, "primary": primary}
        return best
    except Exception:
        logger.debug("Could not load benchmark scores from HF dataset", exc_info=True)
        return {}


def fetch_models(task_filter: str = "any") -> pd.DataFrame:
    try:
        models = list(list_models(filter="colliderml", limit=500, full=True))
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

    scores_by_model = _load_best_scores()

    rows = []
    for m in models:
        tags = m.tags or []
        task = next((t for t in tags if t in ("tracking", "jets", "anomaly")), "general")
        if task_filter != "any" and task != task_filter:
            continue
        bench = scores_by_model.get(m.modelId, {})
        primary_score = bench.get("primary", "")
        bench_task = bench.get("task", "")
        rows.append({
            "model": m.modelId,
            "task": task,
            "benchmark": f"{bench_task}: {primary_score:.4f}" if primary_score != "" else "",
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
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
