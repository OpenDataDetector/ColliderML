# Physics utilities

These utilities operate on ColliderML tables (Polars DataFrame/LazyFrame) and are designed to preserve physical consistency.

## Pileup subsampling

Subsample full-pileup events down to a target number of vertices.

Semantics:

- `vertex_primary` labels vertices as integers `1..N` per event
- To target pileup `K`, remove vertices with `vertex_primary > K`
- Remove particles and calo contributions from removed vertices

```python
from colliderml.physics import subsample_pileup

tables_sub = subsample_pileup(tables, target_vertices=50)
```

## Decay traversal labels

Assign a per-particle label identifying the primary ancestor in the decay chain:

```python
from colliderml.physics import assign_primary_ancestor

particles_labeled = assign_primary_ancestor(frames["particles"])
```

## Calorimeter calibration

Calibration can be applied to calo hit energies (and optionally contribution energies).

### Default ODD calibration

```python
from colliderml.physics.calibration import apply_calo_calibration, odd_default_calo_calibration

calo_cal = apply_calo_calibration(frames["calo_hits"], odd_default_calo_calibration())
```

### YAML-based calibration

You can also provide a YAML mapping keyed by detector IDs or region names:

```yaml
detector_scale:
  ecal_barrel: 37.5
  ecal_endcap: 38.7
  hcal_barrel: 45.0
  hcal_endcap: 46.9
default_scale: 1.0
apply_to_contrib: true
```

```python
from colliderml.physics.calibration import apply_calo_calibration

calo_cal = apply_calo_calibration(frames["calo_hits"], "calibration.yaml")
```


