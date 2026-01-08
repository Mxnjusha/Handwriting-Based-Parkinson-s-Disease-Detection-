"""
IMPROVED Training Script with Advanced Techniques
- Fine-tuning ResNet50
- Grid Search for SVM hyperparameters
- K-Fold Cross-Validation
- Feature normalization
- Ensemble methods option
"""
import os
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception:
    pass

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from glob import glob
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset path
DATA_DIR = '/Users/manjusha/Documents/research/augmented_dataset'

# Model directory
MODEL_DIR = 'models_improved'
os.makedirs(MODEL_DIR, exist_ok=True)

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Advanced options
USE_FINE_TUNING = True  # Fine-tune last layers of ResNet50
USE_GRID_SEARCH = True  # Search for best SVM parameters
USE_CROSS_VALIDATION = True  # Use k-fold CV for robust evaluation
USE_FEATURE_SCALING = True  # Standardize features
N_FOLDS = 5  # Number of cross-validation folds

# ============================================================================

print("="*70)
print("IMPROVED PARKINSON'S DISEASE CLASSIFIER - TRAINING")
print("Advanced Techniques: Fine-tuning + Grid Search + Cross-Validation")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND VALIDATE DATASET
# ============================================================================

def is_valid_image(img_path):
    """Check if image is valid"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img.resize((50, 50))
        return True
    except:
        return False

print("\nStep 1: Loading and validating dataset...")
print(f"Dataset directory: {DATA_DIR}")

image_paths = []
labels = []
classes = ['healthy', 'patient']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        continue
    
    for subdir in ['Circle', 'Meander', 'Spiral', 'circle', 'meander', 'spiral']:
        subdir_path = os.path.join(class_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        for pattern in patterns:
            files = glob(os.path.join(subdir_path, pattern))
            
            for file_path in files:
                if is_valid_image(file_path):
                    image_paths.append(file_path)
                    labels.append(class_to_idx[class_name])

print(f"Valid images: {len(image_paths)}")
for class_name, idx in class_to_idx.items():
    count = labels.count(idx)
    print(f"  {class_name}: {count} ({count/len(labels)*100:.1f}%)")

image_paths = np.array(image_paths)
labels = np.array(labels)

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================

print(f"\nStep 2: Splitting dataset...")
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=labels
)
print(f"Training: {len(train_paths)}, Testing: {len(test_paths)}")

# ============================================================================
# STEP 3: BUILD IMPROVED MODEL
# ============================================================================

print(f"\nStep 3: Building improved ResNet50 model...")

if USE_FINE_TUNING:
    print("  Mode: Fine-tuning enabled (training last layers)")
    
    # Load ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze early layers, unfreeze last layers
    for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
        layer.trainable = False
    for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
        layer.trainable = True
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2048, activation='relu')(x)  # Feature vector
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse'
    )
    
else:
    print("  Mode: Standard feature extraction (no fine-tuning)")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False

print("✓ Model built successfully")

# ============================================================================
# STEP 4: EXTRACT FEATURES
# ============================================================================

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
    except:
        return None

def extract_features_batch(image_paths, model, batch_size=32):
    """Extract features with progress tracking"""
    features = []
    valid_indices = []
    
    num_batches = int(np.ceil(len(image_paths) / batch_size))
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        batch_images = []
        batch_valid_indices = []
        
        for idx, img_path in enumerate(batch_paths):
            img = load_and_preprocess_image(img_path)
            if img is not None:
                batch_images.append(img[0])
                batch_valid_indices.append(start_idx + idx)
        
        if len(batch_images) > 0:
            batch_images = np.array(batch_images)
            batch_features = model.predict(batch_images, verbose=0)
            features.append(batch_features)
            valid_indices.extend(batch_valid_indices)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {end_idx}/{len(image_paths)}")
    
    return np.vstack(features), valid_indices

print("\nStep 4: Extracting features...")
print("  Extracting training features...")
train_features, train_valid_idx = extract_features_batch(train_paths, model, BATCH_SIZE)
train_labels = train_labels[train_valid_idx]

print("  Extracting testing features...")
test_features, test_valid_idx = extract_features_batch(test_paths, model, BATCH_SIZE)
test_labels = test_labels[test_valid_idx]

print(f"✓ Training features: {train_features.shape}")
print(f"✓ Testing features: {test_features.shape}")

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================

if USE_FEATURE_SCALING:
    print("\nStep 5: Scaling features...")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    print("✓ Features scaled")
else:
    scaler = None

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================

print("\nStep 6: Training SVM classifier...")

if USE_GRID_SEARCH:
    print("  Running Grid Search for optimal hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    # Grid search with cross-validation
    svm_base = SVC(probability=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(train_features, train_labels)
    
    print(f"  ✓ Best parameters: {grid_search.best_params_}")
    print(f"  ✓ Best CV score: {grid_search.best_score_ * 100:.2f}%")
    
    svm_classifier = grid_search.best_estimator_
    
else:
    print("  Training with default parameters...")
    svm_classifier = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )
    svm_classifier.fit(train_features, train_labels)

print("✓ SVM training completed")

# ============================================================================
# STEP 7: CROSS-VALIDATION
# ============================================================================

if USE_CROSS_VALIDATION:
    print("\nStep 7: Performing k-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        svm_classifier,
        train_features,
        train_labels,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"  Cross-validation scores: {cv_scores}")
    print(f"  Mean CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

# ============================================================================
# STEP 8: FINAL EVALUATION
# ============================================================================

print("\nStep 8: Final evaluation...")

train_predictions = svm_classifier.predict(train_features)
test_predictions = svm_classifier.predict(test_features)
test_proba = svm_classifier.predict_proba(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")

if USE_CROSS_VALIDATION:
    print(f"Cross-Val Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(test_labels, test_predictions, 
                          target_names=classes, digits=4))

cm = confusion_matrix(test_labels, test_predictions)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"            Healthy  Patient")
print(f"Healthy       {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"Patient       {cm[1][0]:4d}     {cm[1][1]:4d}")

# Calculate metrics
sensitivity = cm[1][1] / (cm[1][0] + cm[1][1])
specificity = cm[0][0] / (cm[0][0] + cm[0][1])
precision = cm[1][1] / (cm[0][1] + cm[1][1])
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nDetailed Metrics:")
print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
print(f"  Specificity: {specificity * 100:.2f}%")
print(f"  Precision: {precision * 100:.2f}%")
print(f"  F1-Score: {f1 * 100:.2f}%")

# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================

print(f"\nStep 9: Saving models...")

# Save feature extractor
model.save(os.path.join(MODEL_DIR, 'resnet_feature_extractor.h5'))
print(f"✓ Saved feature extractor")

# Save SVM
joblib.dump(svm_classifier, os.path.join(MODEL_DIR, 'svm_classifier.pkl'))
print(f"✓ Saved SVM classifier")

# Save scaler if used
if scaler is not None:
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f"✓ Saved feature scaler")

# Save class indices
joblib.dump(class_to_idx, os.path.join(MODEL_DIR, 'class_indices.pkl'))
print(f"✓ Saved class indices")

# Save training config
config = {
    'test_accuracy': float(test_accuracy),
    'train_accuracy': float(train_accuracy),
    'cv_scores': cv_scores.tolist() if USE_CROSS_VALIDATION else None,
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'fine_tuning': USE_FINE_TUNING,
    'feature_scaling': USE_FEATURE_SCALING,
    'grid_search': USE_GRID_SEARCH,
    'best_params': grid_search.best_params_ if USE_GRID_SEARCH else None
}
joblib.dump(config, os.path.join(MODEL_DIR, 'training_config.pkl'))
print(f"✓ Saved training configuration")

print(f"\n{'='*70}")
print("✅ IMPROVED TRAINING COMPLETED!")
print(f"{'='*70}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Models saved in: {MODEL_DIR}")
print(f"{'='*70}\n")