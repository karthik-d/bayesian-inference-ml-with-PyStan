# Probabilistic Machine Learning Experiments with PyStan

> **Note that** the the first run of all `*/infer.py` files will be slow since the PM model will be built and stored as `pickle` files. Subsequent runs will reuse this `pickle` file.

> **Be sure to** remove or relocate the corresponding `*.pkl` file(s) when changing model configurations. The older model will be used for inference, otherwise.