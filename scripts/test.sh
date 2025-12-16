export PYTHONPATH=$PYTHONPATH:.
pyclean . -d
python -m unittest tests.test_flow_metrics
python -m unittest tests.test_detection_metrics
python -m unittest tests.test_computation
python -m unittest tests.test_feature_metrics
pyclean . -d