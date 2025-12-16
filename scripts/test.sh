export PYTHONPATH=$PYTHONPATH:.
python -m unittest tests.test_flow_metrics
python -m unittest tests.test_accuracy