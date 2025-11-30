# test_simple.py
print("🚀 TEST: Running simple test")
import tensorflow as tf
print("✅ TensorFlow imported")

if tf.keras.models.load_model('ecg_heart_disease_model.h5'):
    print("✅ Model loaded")
else:
    print("❌ Model not loaded")