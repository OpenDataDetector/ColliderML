---
title: Detector & Events
---

# The detector & a real event

ColliderML is full-detail simulation of the **Open Data Detector (ODD)** — a
realistic, openly-available HL-LHC-class detector. Below: the detector itself,
then a live, interactive view of a genuine simulated event from the dataset.

## The Open Data Detector

![The full Open Data Detector — tracker, calorimeters and muon system in a cutaway view](/detector/detector-full.jpg)

*The full ODD: silicon tracker at the core, electromagnetic and hadronic
calorimeters (cream), surrounded by the muon system.*

![The ODD silicon tracker — pixel and strip layers, barrel and endcaps](/detector/detector-tracker.jpg)

*The silicon tracker — pixel and short/long-strip layers across barrel and
endcaps. This is where `tracker_hits` are recorded.*

![The ODD calorimeter and solenoid in cross-section](/detector/detector-calo.jpg)

*The calorimeter and solenoid in cross-section — the EM and hadronic
calorimeters that produce the `calo_hits` energy deposits.*

## Explore a simulated event

A real **ttbar** event from `ttbar_pu0`, rendered in your browser: reconstructed
**tracks**, calorimeter **energy deposits**, raw **tracker hits**, and clustered
**jets** (anti-k<sub>T</sub>, R = 0.4). Drag to orbit, scroll to zoom, and use the
controls to switch events and toggle layers.

<EventDisplay />

::: tip Powered by hep-viz
The event display embeds [**hep-viz**](https://github.com/FinnbarWilson/hep-viz)
(F. Wilson & G. Facini, UCL — [10.5281/zenodo.18387794](https://doi.org/10.5281/zenodo.18387794),
MIT), a Three.js HEP event display built for ColliderML data. Here it runs fully
client-side over pre-exported events — no backend required. To explore the whole
dataset interactively, `pip install hep-viz` and run `hep-viz view <data-dir>`.
:::

## Citing hep-viz

If you use the event display in your work, please cite hep-viz:

```bibtex
@software{hepviz,
  author    = {Wilson, Finnbar and Facini, Gabriel},
  title     = {hep-viz},
  version   = {0.1.5},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.18387794},
  url       = {https://doi.org/10.5281/zenodo.18387794}
}
```

::: info Credits
Detector renders: the [Open Data Detector](https://github.com/OpenDataDetector/OpenDataDetector) project.
:::
