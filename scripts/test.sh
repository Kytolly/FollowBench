export PYTHONPATH=$PYTHONPATH:.
python -m unittest tests.test_flow_metrics
python -m unittest tests.test_detection_metrics
python -m unittest tests.test_computation