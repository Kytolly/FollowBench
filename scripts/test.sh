#!/bin/bash
# scripts/test.sh

export PYTHONPATH=$PYTHONPATH:.
pyclean . -d

echo ">>> Running Metric Tests..."
python -m unittest tests.test_metrics
echo ">>> Metric Tests Completed!"

echo ">>> Uploading Dataflow to Repo..."
python tests/test_dataflow.py
python scripts/upload.py -e test
echo ">>> Dataflow OK!"

pyclean . -d