# AIS25: ColliderML Poster

## Conference Contribution

**Title:** ColliderML: Enabling Foundation Models in High Energy Physics Through Low-Level Detector Data

**Authors:** Daniel Murnane, Paul Gessinger-Befurt, Andreas Salzburger, Anna Zaborowska

**Event:** AI in Science Summit 2025 (AIS25)  
**Dates:** November 3–4, 2025  
**Location:** Bella Center Copenhagen, Denmark

If you visited our poster during AIS25, thank you for your interest! For more information about the conference, please visit the [AIS25 website](https://www.ais25.eu/).

## Abstract

ColliderML introduces an open dataset of one million fully simulated proton-proton collisions under High-Luminosity Large Hadron Collider (HL-LHC) conditions. Unlike existing fast-simulation datasets operating on high-level objects, ColliderML provides detector-level measurements across ten physics processes, including hits, energy deposits, and reconstructed tracks from realistic detector geometry under high luminosity pile-up conditions (µ ≈ 200). This work argues that foundation models trained on such low-level data represent the future of collider physics, positioning ColliderML as the infrastructure to realize this vision.

## Getting the Data

The ColliderML dataset is available through a lightweight library, accessing a NERSC Public Portal. For instructions on downloading and using the data, please visit the [ColliderML homepage](https://opendatadetector.github.io/ColliderML/).

## Acknowledgments

This work is made possible by a generous NERSC computing allocation: This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award HEP-ERCAP0034031.

DM is supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429)

## Bugs and Feedback

If you encounter any bugs or have any feedback, please [open an issue](https://github.com/OpenDataDetector/ColliderML/issues) on the GitHub repository. You can also contact [daniel.thomas.murnane@cern.ch](mailto:daniel.thomas.murnane@cern.ch).

## References

The below references are cited in the ColliderML AIS25 contribution.

[1]  S. Amrouche et al., "**The Tracking Machine Learning Challenge: Accuracy Phase,**" arXiv preprint, arXiv:1904.06778 (2019). \
[2]  J. Alwall et al., "**The automated computation of tree-level and next-to-leading order differential cross sections, and their matching to parton shower simulations,**" *JHEP* 07, 079 (2014). \
[3]  T. Sjöstrand et al., "**An Introduction to PYTHIA 8.2,**" *Comput. Phys. Commun.* 191, 159–177 (2015). \
[4]  S. Höche et al., "**Vector-boson fusion at next-to-leading order QCD with parton showers,**" *SciPost Phys.* 12, 091 (2022). \
[5]  ATLAS Collaboration, "**ATLAS ITk Track Reconstruction with a GNN-based pipeline,**" ATL-ITK-PROC-2022-006 (2022). \
[6]  P. Gessinger-Befurt et al., "**The Open Data Detector Tracking System,**" presented at Instrumentation Days (IN2P3) 2023. \
[7]  M. Bacchetta et al., "**CLD --- A Detector Concept for the FCC-ee,**" arXiv:1911.12230 (2019). \
[8]  C. Adloff et al. (CALICE Collaboration), "**Construction and commissioning of the CALICE analog hadron calorimeter prototype,**" *JINST* 5, P05004 (2010). \
[9]  H. Aihara, P. Burrows, M. Oreglia et al. (SiD Collaboration), "**SiD Letter of Intent,**" arXiv:0911.0006 (2009). \
[10] CMS Collaboration, "**The Phase-2 Upgrade of the CMS Endcap Calorimeter,**" CERN-LHCC-2017-023, CMS-TDR-019 (2017). \
[11] F. Gaede et al., "**EDM4hep: A common event data model for HEP,**" *EPJ Web Conf.* 251, 03026 (2021). \
[12] F. Gaede et al., "**The DD4hep detector description toolkit,**" *EPJ Web Conf.* 245, 02004 (2020). \
[13] S. Agostinelli et al. (GEANT4 Collaboration), "**GEANT4—A Simulation Toolkit,**" *Nucl. Instrum. Meth. A* 506, 250–303 (2003). \
[14] ACTS Collaboration, "**A Common Tracking Software (ACTS),**" *EPJ Web Conf.* 245, 02028 (2020). \
[15] M. Brondolin et al., "**The Key4HEP Software Stack: Recent Progress,**" *EPJ Web Conf.* 295, 05010 (2024). \
[16] Key4HEP Collaboration, "**k4DetPerformance: CLD/Key4HEP Reconstruction and Digitisation Examples,**" (2023), [https://github.com/key4hep/k4DetPerformance](https://github.com/key4hep/k4DetPerformance). \
[17] J. Gao et al., "**Track Reconstruction with the ACTS Combinatorial Kalman Filter and Seeding,**" arXiv:2311.00241 (2023). \
[18] ATLAS Collaboration, "**Topological cell clustering in the ATLAS calorimeters and its performance in LHC Run 1,**" *Eur. Phys. J. C* 77, 490 (2017). \
[19] X. Ju et al., "**Performance of a geometric deep learning pipeline for HL-LHC particle tracking,**" Eur. Phys. J. C 81, 876 (2021). \
[20] J. Duarte et al. (Exa.TrkX Collaboration), "**End-to-End Particle Tracking and Reconstruction with GNNs at the HL-LHC,**" arXiv preprint, arXiv:2203.08800 (2022). \
[21] ATLAS Collaboration, "**ATLAS ITk Track Reconstruction with a GNN-based pipeline,**" ATL-ITK-PROC-2022-006 (2022). \
[22] S. Caillou et al., "**Physics Performance of the ATLAS GNN4ITk Track Reconstruction Chain,**" EPJ Web of Conf. 295, 03030 (2024). \
[23] ATLAS Collaboration, "**Technical Design Report for the ATLAS High-Granularity Timing Detector (HGTD),**" CERN-LHCC-2020-007, ATLAS-TDR-031 (2020). \
[24] CERN LCG, "**LCG Views and Releases (documentation page),**" (2025), [https://lcginfo.cern.ch/](https://lcginfo.cern.ch/).

## Citation

If you use the ColliderML dataset in your research, please cite:
```bibtex
@conference{colliderml-ais25,
  title={ColliderML: Enabling Foundation Models in High Energy Physics Through Low-Level Detector Data},
  author={Murnane, Daniel and Gessinger-Befurt, Paul and Salzburger, Andreas and Zaborowska, Anna},
  booktitle={AI in Science Summit 2025 (AIS25)},
  year={2025}
}
```

