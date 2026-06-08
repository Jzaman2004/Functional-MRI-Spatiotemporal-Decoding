#!/usr/bin/env bash
set -euo pipefail

# Lightweight runner for the fMRI pipeline. Edit dataset path inside `src/fMRI_code.py` if needed.
python3 -u src/fMRI_code.py
