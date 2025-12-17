#!/bin/bash
# scripts/test.sh

export PYTHONPATH=$PYTHONPATH:.
pyclean . -d

echo ">>> Running Metric Tests..."
python -m unittest tests.test_metrics
echo ">>> Metric Tests Completed!"

pyclean . -d