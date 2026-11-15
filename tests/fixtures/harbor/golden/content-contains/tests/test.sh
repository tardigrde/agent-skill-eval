#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
export PYTHONPATH=/tests
python /tests/verifier.py
