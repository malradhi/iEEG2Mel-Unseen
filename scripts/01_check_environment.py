"""
Check Python packages and GPU availability.

This is a lightweight sanity check before running extraction/training.
"""

import os
import sys
import importlib

print("=" * 70)
print("Environment check")
print("=" * 70)
print("Current directory:", os.getcwd())
print("Python version:", sys.version)

for name in ["numpy", "pandas", "sklearn", "matplotlib", "scipy", "pynwb"]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "version not available")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: NOT AVAILABLE ({exc})")

try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
    print("GPUs detected:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("tensorflow: NOT AVAILABLE or failed to import:", exc)

print("=" * 70)
print("Check completed.")
