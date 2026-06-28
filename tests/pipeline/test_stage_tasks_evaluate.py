"""Pipeline stage: tasks evaluate (local preview scoring).

GREEN: registry + schema validation + metric dict. RED: jets/anomaly local
scoring is not grounded in canonical truth — jets pairs predictions to a
fixed-RNG truth positionally, and anomaly derives truth from the submitter's
own ``channel`` column. Both are encoded as contract assertions that fail now.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import colliderml

pytestmark = pytest.mark.pipeline


EXPECTED_TASKS = {"tracking", "jets", "anomaly", "tracking_latency", "tracking_small", "data_loading"}


def test_registry_lists_all_tasks():
    assert set(colliderml.tasks.list_tasks()) == EXPECTED_TASKS


def test_evaluate_rejects_bad_schema():
    bad = pa.table({"event_id": [1, 2], "jet_id": [0, 1]})  # missing prob_* columns
    with pytest.raises(ValueError):
        colliderml.tasks.evaluate("jets", bad)


def test_evaluate_returns_metric_dict():
    preds = _jets_table(event_ids=list(range(20)))
    scores = colliderml.tasks.evaluate("jets", preds)
    assert set(scores) == {"btag_auc", "light_rej_70", "c_rej_70"}
    assert all(isinstance(v, float) for v in scores.values())


def test_jets_score_is_by_identity_not_row_order():
    """RED: scoring must join predictions to truth by (event_id, jet_id), so
    shuffling the prediction rows must NOT change the score. Today jets pairs
    predictions to a fixed-RNG truth positionally, so a shuffle changes it.
    """
    preds = _jets_table(event_ids=list(range(20)))
    shuffled = preds.take(list(range(19, -1, -1)))  # reverse row order
    a = colliderml.tasks.evaluate("jets", preds)["btag_auc"]
    b = colliderml.tasks.evaluate("jets", shuffled)["btag_auc"]
    assert a == pytest.approx(b), (
        "jets score changed when prediction rows were reordered — it is paired "
        "to truth positionally (fixed RNG) instead of joined by event_id/jet_id "
        "against canonical truth (colliderml/tasks/jets/task.py)."
    )


def test_anomaly_score_ignores_self_supplied_labels():
    """RED: AUROC must use canonical truth (which events are really BSM), not the
    submitter's own ``channel`` column. Today a user controls their own truth, so
    merely relabeling channels flips the AUROC — it must be invariant instead.
    """
    scores = [0.9 - 0.04 * i for i in range(20)]
    eids = list(range(20))
    bsm, sm = "zprime_pu0", "ttbar_pu0"
    # t1: high scores labeled BSM (cheats to AUROC~1); t2: labels reversed (~0)
    t1 = pa.table({"event_id": eids, "anomaly_score": scores,
                   "channel": [bsm] * 10 + [sm] * 10})
    t2 = pa.table({"event_id": eids, "anomaly_score": scores,
                   "channel": [sm] * 10 + [bsm] * 10})
    a = colliderml.tasks.evaluate("anomaly", t1)["auroc"]
    b = colliderml.tasks.evaluate("anomaly", t2)["auroc"]
    assert a == pytest.approx(b), (
        "anomaly AUROC changed when the submitter relabeled the 'channel' column — "
        "truth is taken from user input instead of canonical labels "
        "(colliderml/tasks/anomaly/task.py)."
    )


def _jets_table(event_ids):
    n = len(event_ids)
    prob_b = [0.05 + 0.9 * (i / max(n - 1, 1)) for i in range(n)]
    prob_c = [(1.0 - p) * 0.4 for p in prob_b]
    prob_light = [1.0 - b - c for b, c in zip(prob_b, prob_c)]
    return pa.table({
        "event_id": list(event_ids),
        "jet_id": list(range(n)),
        "prob_b": prob_b,
        "prob_c": prob_c,
        "prob_light": prob_light,
    })
