# test_installation.py
try:
    import fastapi
    import uvicorn
    import tensorflow as tf
    import cv2
    import PIL
    import numpy as np
    import sklearn
    print("✅ All imports successful!")
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"❌ Import error: {e}")