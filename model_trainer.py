"""
Machine Learning Model Training Module
Supports multiple algorithms for credit default prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, Tuple, Optional
from config import *


class CreditModelTrainer:
    """Train and evaluate credit scoring models"""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        self.training_history = {}
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: list,
        target_col: str = 'default_label',
        test_size: float = TEST_SIZE,
        use_smote: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training"""
        
        # Remove any rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Select features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        print(f"Features: {len(available_features)}")
        print(f"Samples: {len(X)}")
        print(f"Default rate: {y.mean():.2%}")
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Store feature names
        self.feature_names = available_features
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Apply SMOTE to handle class imbalance
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Train set: {len(X_train)} samples")
            print(f"After SMOTE - Default rate: {y_train.mean():.2%}")
        
        # Scale features
        print("Scaling features...")
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=available_features
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=available_features
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train Logistic Regression model"""
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = model
        return model
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        tune_hyperparameters: bool = False
    ) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        return model
    
    def train_gradient_boosting(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> GradientBoostingClassifier:
        """Train Gradient Boosting model"""
        print("\nTraining Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = model
        return model
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        self.models['XGBoost'] = model
        return model
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            class_weight='balanced',
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        self.models['LightGBM'] = model
        return model
    
    def evaluate_model(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """Evaluate model performance"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_test: pd.Series,
        models_to_train: list = None
    ) -> pd.DataFrame:
        """Train multiple models and compare performance"""
        
        if models_to_train is None:
            models_to_train = [
                'Logistic Regression',
                'Random Forest', 
                'Gradient Boosting',
                'XGBoost',
                'LightGBM'
            ]
        
        results = []
        
        for model_name in models_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}")
            
            # Train model
            if model_name == 'Logistic Regression':
                model = self.train_logistic_regression(X_train, y_train)
            elif model_name == 'Random Forest':
                model = self.train_random_forest(X_train, y_train)
            elif model_name == 'Gradient Boosting':
                model = self.train_gradient_boosting(X_train, y_train)
            elif model_name == 'XGBoost':
                model = self.train_xgboost(X_train, y_train)
            elif model_name == 'LightGBM':
                model = self.train_lightgbm(X_train, y_train)
            else:
                print(f"Unknown model: {model_name}")
                continue
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results.append(metrics)
            
            print(f"\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        # Select best model
        self.best_model_name = results_df.iloc[0]['model_name']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n{'='*60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
        print(f"{'='*60}")
        
        return results_df
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from best model"""
        
        if self.best_model is None:
            print("No model trained yet!")
            return pd.DataFrame()
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("Model doesn't support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
    
    def save_model(self, model_dir: str = MODELS_DIR):
        """Save trained model and scaler"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        model_path = os.path.join(model_dir, TRAINED_MODEL_FILE)
        joblib.dump(self.best_model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, SCALER_FILE)
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Save feature names
        feature_path = os.path.join(model_dir, 'feature_names.pkl')
        joblib.dump(self.feature_names, feature_path)
        print(f"✓ Feature names saved to {feature_path}")
        
        # Save feature importance
        feature_imp = self.get_feature_importance(top_n=50)
        imp_path = os.path.join(model_dir, FEATURE_IMPORTANCE_FILE)
        feature_imp.to_csv(imp_path, index=False)
        print(f"✓ Feature importance saved to {imp_path}")
    
    def load_model(self, model_dir: str = MODELS_DIR):
        """Load saved model and scaler"""
        
        model_path = os.path.join(model_dir, TRAINED_MODEL_FILE)
        scaler_path = os.path.join(model_dir, SCALER_FILE)
        feature_path = os.path.join(model_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_path)
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
        print(f"✓ Feature names loaded ({len(self.feature_names)} features)")


if __name__ == "__main__":
    from data_generator import generate_and_save_data
    from feature_engineering import FeatureEngineer
    
    # Generate data if needed
    if not os.path.exists(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"):
        print("Generating synthetic data...")
        df = generate_and_save_data()
    else:
        print("Loading existing data...")
        df = pd.read_csv(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    engineer = FeatureEngineer()
    df = engineer.prepare_for_modeling(df)
    
    # Get features
    feature_cols = engineer.get_all_model_features()
    
    # Train models
    trainer = CreditModelTrainer()
    
    print("\nPreparing data for training...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, feature_cols, use_smote=True
    )
    
    print("\nTraining models...")
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    print("\n\nModel Comparison:")
    print(results[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])
    
    print("\n\nTop 10 Important Features:")
    print(trainer.get_feature_importance(top_n=10))
    
    # Save model
    print("\nSaving best model...")
    trainer.save_model()
