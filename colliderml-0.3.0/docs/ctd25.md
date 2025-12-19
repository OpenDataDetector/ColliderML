# CTD25: ColliderML Poster

## Conference Contribution

**Title:** The Impact of Low-Level Detector Data on Physics Reconstruction Performance

**Authors:** Daniel Murnane, Paul Gessinger, Andreas Salzburger, Anna Zaborowska

**Event:** Connecting the Dots 2025 (CTD25)

**Dates:** November 10–14, 2025

**Location:** Tokyo, Japan

If you visited our poster during CTD25, thank you for your interest! For more information about the conference, please visit the [CTD25 website](https://indico.cern.ch/event/1499357/).

## Abstract

With the recent release of the OpenDataDetector High-Luminosity Physics Benchmark Dataset (aka ColliderML), it is now feasible to study the behaviour of machine learning (ML) algorithms in a variety of full simulation conditions. We present a suite of pilot studies that establish the utility of such a multi-scale, full-detector dataset for ML development. First, we examine the ability of models to distinguish between different physics channels from low- and/or high-level objects, and to harness multiple detector regions in reconstruction (for example combining tracker hits and calorimeter clusters to perform track reconstruction and particle flow on low-level readout). Second, we study generalizability to unseen SM and BSM physics conditions. Third, we test whether symmetry-preserving architectures (e.g. Lorentz equivariance) continue to work well in cases where symmetry may be broken by the detector effects of digitized full simulation. Finally, we broadly study the potential of this dataset as a platform for building true particle physics foundation models: systems that can directly consume low level data and be fine-tuned to a variety of downstream reconstruction and analysis tasks.

## Getting the Data

The ColliderML dataset is available through a lightweight library, accessing a NERSC Public Portal. For instructions on downloading and using the data, please visit the [ColliderML homepage](https://opendatadetector.github.io/ColliderML/).

## Acknowledgments

This work is made possible by a generous NERSC computing allocation: This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award HEP-ERCAP0034031.

DM is supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429)

## Bugs and Feedback

If you encounter any bugs or have any feedback, please [open an issue](https://github.com/OpenDataDetector/ColliderML/issues) on the GitHub repository. You can also contact [daniel.thomas.murnane@cern.ch](mailto:daniel.thomas.murnane@cern.ch).

## References

The below references are cited in the ColliderML CTD25 contribution.

[1]  ATLAS Collaboration, "**Transforming Jet Flavour Tagging at ATLAS,**" arXiv:2505.19689, CERN-EP-2025-103 (2025). \
[2]  CMS Collaboration, "**Run 3 performance and advances in heavy-flavor jet tagging in CMS,**" arXiv:2412.05863 (2024). \
[3]  J. de Favereau et al., "**DELPHES 3: a modular framework for fast simulation of a generic collider experiment,**" *JHEP* 02, 057 (2014). \
[4]  ATLAS Collaboration, "**ATLAS releases 65 TB of open data for research,**" July 2024, [https://opendata.atlas.cern](https://opendata.atlas.cern). \
[5]  H. Qu, C. Li, and S. Qian, "**Particle Transformer for Jet Tagging,**" *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*, arXiv:2202.03772 (2022). \
[6]  A. Salzburger, A. Zaborowska, D. T. Murnane, M.-T. Pham, and P. Gessinger, "**ColliderML: The First Release of an OpenDataDetector High-Luminosity Physics Benchmark Dataset,**" ACAT 2025, September 8–12, 2025, Hamburg, Germany, [https://indico.cern.ch/event/1488410/contributions/6561432/](https://indico.cern.ch/event/1488410/contributions/6561432/). \
[7]  ACTS Collaboration, "**A Common Tracking Software (ACTS),**" *EPJ Web Conf.* 245, 02028 (2020). \
[8]  M. Cacciari, G. P. Salam, and G. Soyez, "**FastJet User Manual,**" *Eur. Phys. J. C* 72, 1896 (2012), arXiv:1111.6097. \
[9]  M. Frank, F. Gaede, C. Grefe, and P. Mato, "**DD4hep: A Detector Description Toolkit for High Energy Physics Experiments,**" *J. Phys.: Conf. Ser.* 513, 022010 (2014). \
[10] S. Agostinelli et al. (GEANT4 Collaboration), "**GEANT4—A Simulation Toolkit,**" *Nucl. Instrum. Meth. A* 506, 250–303 (2003).

## Citation

If you use the ColliderML dataset in your research, please cite:
```bibtex
@conference{colliderml-ctd25,
  title={The Impact of Low-Level Detector Data on Physics Reconstruction Performance},
  author={Murnane, Daniel and Gessinger, Paul and Salzburger, Andreas and Zaborowska, Anna},
  booktitle={Connecting the Dots 2025 (CTD25)},
  year={2025},
  month={November},
  address={Tokyo, Japan},
  url={https://indico.cern.ch/event/1499357/}
}
```

