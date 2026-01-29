"""
Random Forest ML Model for Dengue Prediction
Trains on the provided dataset and makes predictions based on blood sample data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ========================
# PATHS
# ========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DENGUE_DATASET = os.path.join(CURRENT_DIR, "dengue_augmented_combined.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "dengue_model.pkl")
FEATURE_NAMES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")

# ========================
# LOAD AND PREPARE DATA
# ========================
def load_and_prepare_data():
    """Load CSV and prepare data for training"""
    try:
        df = pd.read_csv(DENGUE_DATASET)
        print(f"✅ Dataset loaded: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DENGUE_DATASET}")
        return None

# ========================
# TRAIN MODEL
# ========================
def train_model(df):
    """Train Random Forest classifier"""
    
    # Prepare features and target
    target_col = None
    for col in df.columns:
        if col.lower() in ['outcome', 'target', 'diagnosis', 'dengue', 'class']:
            target_col = col
            break
    
    if target_col is None:
        print("❌ Could not find target column (Outcome/Target/Diagnosis)")
        return None, None, None
    
    print(f"\n🎯 Target column: {target_col}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    
    print(f"\n📊 Features after encoding: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    # Train-test split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Create and train Random Forest model with pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\n🤖 Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Training Complete!")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = model.named_steps['rf'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(importance_df.head(10))
    
    return model, feature_names, {
        'accuracy': float(accuracy),
        'n_samples': len(df),
        'features': len(feature_names),
        'classes': list(y.unique())
    }

# ========================
# SAVE MODEL
# ========================
def save_model(model, feature_names):
    """Save trained model and feature names"""
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n💾 Model saved: {MODEL_PATH}")
        
        with open(FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"💾 Feature names saved: {FEATURE_NAMES_PATH}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

# ========================
# LOAD MODEL
# ========================
def load_model():
    """Load saved model and feature names"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
            print("⚠️ Model files not found. Training new model...")
            df = load_and_prepare_data()
            if df is None:
                return None, None
            model, feature_names, _ = train_model(df)
            if model is None:
                return None, None
            save_model(model, feature_names)
        else:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"✅ Model loaded from {MODEL_PATH}")
        
        return model, feature_names
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# ========================
# MAKE PREDICTION
# ========================
def predict_dengue(blood_data_dict, feature_names, model):
    """
    Make dengue prediction from blood sample data
    
    Args:
        blood_data_dict: Dictionary with blood parameter values
        feature_names: List of feature names used during training
        model: Trained RandomForest model
    
    Returns:
        dict with prediction result and confidence
    """
    try:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([blood_data_dict])
        
        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure all features are present (missing ones are filled with 0)
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Keep only the features used during training
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        return {
            'prediction': str(prediction),
            'confidence': float(max(prediction_proba) * 100),
            'probabilities': {
                str(model.classes_[i]): float(prob * 100) 
                for i, prob in enumerate(prediction_proba)
            }
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            'error': str(e),
            'prediction': 'ERROR'
        }

# ========================
# MAIN - Train if needed
# ========================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 RANDOM FOREST DENGUE PREDICTION MODEL")
    print("=" * 60)
    
    df = load_and_prepare_data()
    if df is not None:
        model, feature_names, info = train_model(df)
        if model is not None:
            save_model(model, feature_names)
            print("\n" + "=" * 60)
            print("✅ MODEL READY FOR DEPLOYMENT")
            print("=" * 60)
