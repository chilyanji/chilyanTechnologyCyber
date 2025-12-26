"""
Machine Learning Model Training - Random Forest & TensorFlow
Compares traditional ML (Random Forest) with deep learning (TensorFlow/Keras)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class PhishingDetectionModel:
    """Train and evaluate phishing detection models"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.tf_model = None
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        print("Preparing data for model training...")
        
        # Separate features and labels
        # Drop non-numeric columns
        self.X = self.df.drop(['url', 'label'], axis=1)
        self.y = self.df['label']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Feature columns: {list(self.X.columns)}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        print(f"Class distribution (test): {np.bincount(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, n_estimators=100):
        """
        Train Random Forest Classifier
        
        Why Random Forest?
        - Ensemble method combining multiple decision trees
        - Handles non-linear relationships well
        - Provides feature importance rankings
        - Robust to outliers and missing values
        - Fast inference for real-time detection
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print(f"Training Random Forest with {n_estimators} trees...")
        self.rf_model.fit(self.X_train, self.y_train)
        print("Training complete!")
        
        # Predictions
        y_pred_train = self.rf_model.predict(self.X_train)
        y_pred_test = self.rf_model.predict(self.X_test)
        y_pred_proba_test = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self.results['rf'] = self._evaluate_model(
            self.y_train, self.y_test, y_pred_train, y_pred_test, y_pred_proba_test, 'Random Forest'
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
        plt.xlabel('Importance Score')
        plt.title('Top 10 Feature Importance - Random Forest')
        plt.tight_layout()
        plt.savefig('output/03_feature_importance_rf.png', dpi=300, bbox_inches='tight')
        print("Saved: output/03_feature_importance_rf.png")
        
        return self.rf_model
    
    def train_tensorflow_model(self, epochs=50, batch_size=16):
        """
        Train TensorFlow/Keras Neural Network
        
        Why TensorFlow?
        - Google's production-grade ML framework
        - GPU acceleration support
        - State-of-the-art for complex patterns
        - Scalable to large datasets
        - Integration with Google Cloud services
        """
        print("\n" + "="*60)
        print("TRAINING TENSORFLOW/KERAS NEURAL NETWORK")
        print("="*60)
        
        input_dim = self.X_train_scaled.shape[1]
        
        # Build model
        self.tf_model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile with weighted loss for imbalanced data
        self.tf_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print("Model Architecture:")
        self.tf_model.summary()
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        history = self.tf_model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight={0: 1, 1: 2}  # Weight positive class more
        )
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Epochs')
        axes[0].legend()
        axes[0].grid()
        
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy Over Epochs')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.savefig('output/04_training_history.png', dpi=300, bbox_inches='tight')
        print("Saved: output/04_training_history.png")
        
        # Predictions
        y_pred_test_proba = self.tf_model.predict(self.X_test_scaled)
        y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
        y_pred_train_proba = self.tf_model.predict(self.X_train_scaled)
        y_pred_train = (y_pred_train_proba > 0.5).astype(int).flatten()
        
        # Evaluate
        self.results['tensorflow'] = self._evaluate_model(
            self.y_train, self.y_test, y_pred_train, y_pred_test, y_pred_test_proba.flatten(), 'TensorFlow Neural Network'
        )
        
        return self.tf_model, history
    
    def _evaluate_model(self, y_train, y_test, y_pred_train, y_pred_test, y_pred_proba_test, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{model_name} - EVALUATION RESULTS:")
        print("-" * 60)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        
        print(f"Training Accuracy:  {train_acc:.4f}")
        print(f"Test Accuracy:      {test_acc:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC Score:      {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        # Save with unique name
        filename = f"output/05_confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Legitimate', 'Phishing']))
        
        # Store results
        return {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba_test
        }
    
    def compare_models(self):
        """Compare Random Forest and TensorFlow models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Metric': ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Random Forest': [
                self.results['rf']['test_accuracy'],
                self.results['rf']['precision'],
                self.results['rf']['recall'],
                self.results['rf']['f1_score'],
                self.results['rf']['roc_auc']
            ],
            'TensorFlow': [
                self.results['tensorflow']['test_accuracy'],
                self.results['tensorflow']['precision'],
                self.results['tensorflow']['recall'],
                self.results['tensorflow']['f1_score'],
                self.results['tensorflow']['roc_auc']
            ]
        })
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = comparison_df['Metric']
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['Random Forest'], width, label='Random Forest', color='steelblue')
        ax.bar(x + width/2, comparison_df['TensorFlow'], width, label='TensorFlow', color='coral')
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/06_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved: output/06_model_comparison.png")
        
        return comparison_df
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.rf_model, 'models/random_forest_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        self.tf_model.save('models/tensorflow_model.h5')
        
        print("\nModels saved:")
        print("- models/random_forest_model.pkl")
        print("- models/scaler.pkl")
        print("- models/tensorflow_model.h5")

# Main execution
if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load featured dataset
    df = pd.read_csv('data/phishing_features.csv')
    
    # Initialize model training
    model = PhishingDetectionModel(df)
    
    # Prepare data
    model.prepare_data()
    
    # Train models
    rf_model = model.train_random_forest(n_estimators=100)
    tf_model, history = model.train_tensorflow_model(epochs=50, batch_size=16)
    
    # Compare models
    comparison = model.compare_models()
    
    # Save models
    model.save_models()
