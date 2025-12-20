#!/bin/bash
# scripts/test.sh

export PYTHONPATH=$PYTHONPATH:.
export APP_ENV="test"
pyclean . -d

echo ">>> Running Metric Tests..."
python -m unittest tests.test_metrics
echo ">>> Metric Tests Completed!"

echo ">>> Uploading Dataflow to Repo..."
python tests/test_dataflow.py
python scripts/upload.py -e test
echo ">>> Dataflow OK!"

echo ">>> Integretion Test..."
python -m tests.test_bench_core
echo "Bench Core OK!"

pyclean . -d