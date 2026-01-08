"""
Model Module - Handles prediction logic
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib

class ParkinsonsClassifier:
    def __init__(self, model_dir='models'):
        """Initialize the classifier with pre-trained models"""
        self.model_dir = model_dir
        self.feature_extractor = None
        self.svm_classifier = None
        self.class_indices = None
        self.class_names = None
        self.load_models()
    
    def load_models(self):
        """Load the trained ResNet50 and SVM models"""
        try:
            # Load ResNet50 feature extractor
            feature_extractor_path = os.path.join(self.model_dir, 'resnet_feature_extractor.h5')
            self.feature_extractor = load_model(feature_extractor_path)
            print(f"✓ Loaded feature extractor from {feature_extractor_path}")
            
            # Load SVM classifier
            svm_path = os.path.join(self.model_dir, 'svm_classifier.pkl')
            self.svm_classifier = joblib.load(svm_path)
            print(f"✓ Loaded SVM classifier from {svm_path}")
            
            # Load class indices
            class_indices_path = os.path.join(self.model_dir, 'class_indices.pkl')
            self.class_indices = joblib.load(class_indices_path)
            
            # Reverse class indices to get class names
            self.class_names = {v: k for k, v in self.class_indices.items()}
            print(f"✓ Classes: {self.class_names}")
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image array
        """
        # Load image with target size
        img = load_img(image_path, target_size=(224, 224))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Expand dimensions to create batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for ResNet50
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def extract_features(self, image_path):
        """
        Extract features using ResNet50
        Args:
            image_path: Path to the image file
        Returns:
            Feature vector
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Extract features
        features = self.feature_extractor.predict(img_array, verbose=0)
        
        return features
    
    def predict(self, image_path):
        """
        Make prediction on an image
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features
            features = self.extract_features(image_path)
            
            # Get prediction
            prediction = self.svm_classifier.predict(features)[0]
            
            # Get probability scores
            probabilities = self.svm_classifier.predict_proba(features)[0]
            
            # Get class name
            predicted_class = self.class_names[prediction]
            
            # Create result dictionary
            result = {
                'predicted_class': predicted_class,
                'confidence': float(probabilities[prediction]) * 100,
                'probabilities': {
                    self.class_names[i]: float(prob) * 100 
                    for i, prob in enumerate(probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def get_diagnosis(self, predicted_class):
        """
        Get human-readable diagnosis
        Args:
            predicted_class: Predicted class name
        Returns:
            Diagnosis string
        """
        if 'parkinson' in predicted_class.lower():
            return "Parkinson's Disease Detected"
        else:
            return "Healthy Control"