from setuptools import setup, find_packages

setup(
    name="colliderml",
    version="0.4.0rc2",
    description="A modern machine learning library for high-energy physics data analysis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Daniel Murnane",
    author_email="dtmurnane@lbl.gov",
    url="https://github.com/OpenDataDetector/ColliderML",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "datasets>=2.14.0",
        "huggingface_hub>=0.20.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "h5py>=3.10.0",
        "polars>=0.20.0",
        "pyyaml>=6.0.0",
        "pyarrow>=14.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "ruff>=0.1.6",
            "mypy>=1.7.0",
            "matplotlib>=3.5.0",
        ],
        # Local simulation via Docker/Podman — auto-clones the
        # colliderml-production repo for pipeline scripts + configs.
        "sim": [
            "pyyaml>=6.0.0",
        ],
        # Remote simulation client — talks to the SaaS backend service.
        "remote": [
            "requests>=2.28.0",
        ],
        # Benchmark task runner — includes scikit-learn for the reference
        # baselines (BDT, IsolationForest). Task scoring itself only needs
        # numpy + pyarrow which are already in install_requires.
        "tasks": [
            "scikit-learn>=1.3.0",
        ],
        # Convenience meta-extra: everything above plus the dev tools.
        "all": [
            "colliderml[sim,remote,tasks,dev]",
        ],
    },
    include_package_data=True,
    package_data={
        "colliderml.simulate": ["presets.yaml"],
    },
    entry_points={
        "console_scripts": [
            "colliderml=colliderml.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    project_urls={
        "Documentation": "https://opendatadetector.github.io/ColliderML",
        "Source": "https://github.com/OpenDataDetector/ColliderML",
        "Issues": "https://github.com/OpenDataDetector/ColliderML/issues",
    },
) 