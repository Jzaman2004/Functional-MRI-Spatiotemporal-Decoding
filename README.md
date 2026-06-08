# Functional-MRI-Spatiotemporal-Decoding

A reproducible codebase and results repository for: "Spatiotemporal Decoding of Dynamic Risk Decisions using Machine Learning and Functional MRI"

Overview
--------
This repository contains the data-processing and analysis pipeline used to decode risk-taking decisions (Balloon Analog Risk Task) from fMRI data (OpenNeuro ds000001). The implementation uses a PyTorch linear classifier wrapped for Scikit-Learn compatibility, region extraction with the Harvard-Oxford cortical atlas (Nilearn), and event-locked spatiotemporal analyses.

Layout
------
- `src/` — Main pipeline and scripts. See `src/fMRI_code.py` for the end-to-end implementation.
- `fMRI_code.ipynb` — Original Colab notebook with the same pipeline cells.
- `results/` — CSV and JSON summary outputs (group accuracies, statistical metrics, region rankings).
- `figures/` — All PNG figure assets (glass brain, slice maps, dashboards, time-series plots).
- `data/` — Optional local BIDS dataset location (not included in this repository).
- `scripts/` — Utility scripts and convenience runners (e.g., smoke tests).

Quick start
-----------
1. Create and activate a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Obtain the BIDS dataset `ds000001` (OpenNeuro). The pipeline can fetch it using DataLad, or you may download and place it under `data/` then update the path in `src/fMRI_code.py`.

3. Run the pipeline (update dataset path in `src/fMRI_code.py` if needed):

```bash
python src/fMRI_code.py
```

Notes
-----
- The pipeline was developed for Google Colab and uses DataLad/git-annex for data retrieval; running locally may require manual installation of system packages (e.g., `git-annex`).
- For reproducible results, install the exact versions listed in `requirements.txt` or use the included `scripts/smoke_test.py` to verify the Python ML stack.

Contact and citation
--------------------
Author: Jawad Zaman. If you reuse this code, please cite the repository and the OpenNeuro dataset `ds000001`.

Last updated: 2026-06-08