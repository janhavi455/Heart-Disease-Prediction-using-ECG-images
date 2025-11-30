# model_training.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ECGClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        # Using EfficientNet for better performance
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, data_dir):
        images = []
        labels = []
        class_names = ['normal', 'abnormal']  # Fixed class names for 2 folders
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                            img = img / 255.0
                            
                            images.append(img)
                            labels.append(class_idx)
        
        return np.array(images), np.array(labels), class_names
    
    def train(self, data_dir, epochs=5, batch_size=16):
        X, y, class_names = self.prepare_data(data_dir)
        
        print(f"Loaded {len(X)} images")
        print(f"Class distribution: Normal - {np.sum(y==0)}, Abnormal - {np.sum(y==1)}")
        
        # Convert labels to categorical
        y_categorical = tf.keras.utils.to_categorical(y, self.num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=4),
            keras.callbacks.ModelCheckpoint('best_ecg_model.h5', save_best_only=True)
        ]
        
        # Initial training with frozen base
        print("Starting initial training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning
        print("Starting fine-tuning...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tune for fewer epochs
        history_fine = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=3,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final model
        print("Evaluating final model...")
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        
        return history, history_fine, class_names

# Train the model

if __name__ == "__main__":
    classifier = ECGClassifier()
    
    dataset_path = r"C:\Users\janha\OneDrive\Desktop\HEART DISAESE PRED USING ECG IMG\ecg_dataset"
    
    # Quick check: load data only
    X, y, class_names = classifier.prepare_data(dataset_path)
    print(f"Number of images found: {len(X)}")
    print(f"Number of labels found: {len(y)}")
    print(f"Classes detected: {class_names}")
    
    # If counts look correct, start training
    if len(X) == 0:
        print("No images found! Check your dataset path and folder structure.")
    else:
        history, history_fine, class_names = classifier.train(dataset_path, epochs=5, batch_size=16)
        classifier.model.save('ecg_heart_disease_model.h5')
        print("Model training completed and saved!")