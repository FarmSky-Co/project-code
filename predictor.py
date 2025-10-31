"""
Prediction and Evaluation Module
Make predictions on new farmer applications
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List
from config import *
from feature_engineering import FeatureEngineer
from scoring_engine import CreditScoringEngine


class CreditPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_dir: str = MODELS_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.engineer = FeatureEngineer()
        self.scorer = CreditScoringEngine()
        
    def load_model(self):
        """Load trained model, scaler, and feature names"""
        
        model_path = os.path.join(self.model_dir, TRAINED_MODEL_FILE)
        scaler_path = os.path.join(self.model_dir, SCALER_FILE)
        feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Please train a model first using model_trainer.py"
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_path)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Features: {len(self.feature_names)}")
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and preprocessing"""
        
        # Apply feature engineering
        df = self.engineer.prepare_for_modeling(df)
        
        # Select only the features used in training
        available_features = [col for col in self.feature_names if col in df.columns]
        
        if len(available_features) != len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            print(f"Warning: Missing features: {missing}")
            # Fill missing features with 0
            for feat in missing:
                df[feat] = 0
        
        X = df[self.feature_names].copy()
        
        # Handle missing values and infinities
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        return X_scaled
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data"""
        
        if self.model is None:
            self.load_model()
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add to dataframe
        df['ml_prediction'] = predictions
        df['default_probability'] = probabilities
        df['ml_decision'] = df['ml_prediction'].map({
            0: 'Approve (ML)', 
            1: 'Decline (ML)'
        })
        
        return df
    
    def predict_with_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions and calculate credit scores"""
        
        # Calculate credit scores
        df = self.scorer.score_applications(df)
        
        # Make ML predictions
        df = self.predict(df)
        
        # Create combined decision
        def combined_decision(row):
            credit_score = row['final_credit_score']
            ml_pred = row['ml_prediction']
            credit_dec = row['credit_decision']
            
            # Strong approve: High score AND ML predicts repayment
            if credit_score >= 70 and ml_pred == 0:
                return 'Approve'
            
            # Strong decline: Low score AND ML predicts default
            elif credit_score < 50 and ml_pred == 1:
                return 'Decline'
            
            # Score-based decision with ML override for extreme cases
            elif credit_score >= 75:
                # Very high score - approve even if ML is uncertain
                return 'Approve'
            
            elif credit_score < 45:
                # Very low score - decline even if ML is optimistic
                return 'Decline'
            
            # Medium scores or conflicting signals - manual review
            else:
                return 'Manual Review'
        
        df['final_decision'] = df.apply(combined_decision, axis=1)
        
        return df
    
    def predict_single(self, farmer_dict: Dict) -> Dict:
        """Predict for a single farmer application"""
        
        # Convert to DataFrame
        df = pd.DataFrame([farmer_dict])
        
        # Get predictions and scores
        df = self.predict_with_scores(df)
        
        # Extract results with detailed component scores
        result = {
            'farmer_id': df.iloc[0].get('farmer_id', 'N/A'),
            'credit_scores': {
                'agro_score': round(df.iloc[0]['agro_score'], 2),
                'market_score': round(df.iloc[0]['market_score'], 2),
                'repayment_score': round(df.iloc[0]['repayment_score'], 2),
                'final_score': round(df.iloc[0]['final_credit_score'], 2)
            },
            'agro_components': {
                'climate_match': round(df.iloc[0]['climate_match'], 2),
                'soil_match': round(df.iloc[0]['soil_match'], 2),
                'yield_performance': round(df.iloc[0]['yield_performance'], 2),
                'irrigation_score': round(df.iloc[0]['irrigation_score'], 2)
            },
            'market_components': {
                'price_trend_score': round(df.iloc[0]['price_trend_score'], 2),
                'harvest_price_score': round(df.iloc[0]['harvest_price_score'], 2),
                'volatility_score': round(df.iloc[0]['volatility_score'], 2),
                'market_access_score': round(df.iloc[0]['market_access_score'], 2)
            },
            'repayment_components': {
                'income_coverage_score': round(df.iloc[0]['income_coverage_score'], 2),
                'transaction_health_score': round(df.iloc[0]['transaction_health_score'], 2),
                'savings_pattern_score': round(df.iloc[0]['savings_pattern_score'], 2),
                'social_capital_score': round(df.iloc[0]['social_capital_score'], 2)
            },
            'ml_prediction': {
                'default_probability': round(df.iloc[0]['default_probability'], 4),
                'prediction': 'Default' if df.iloc[0]['ml_prediction'] == 1 else 'Repay',
                'decision': df.iloc[0]['ml_decision']
            },
            'decisions': {
                'rule_based': df.iloc[0]['credit_decision'],
                'ml_based': df.iloc[0]['ml_decision'],
                'final': df.iloc[0]['final_decision']
            }
        }
        
        return result
    
    def batch_predict(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Make predictions on a batch of applications"""
        
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Processing {len(df)} applications...")
        df = self.predict_with_scores(df)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Predictions saved to {output_file}")
        
        # Summary
        print("\nPrediction Summary:")
        print(f"Total applications: {len(df)}")
        print(f"ML Predicted defaults: {(df['ml_prediction'] == 1).sum()}")
        print(f"ML Predicted repayments: {(df['ml_prediction'] == 0).sum()}")
        print(f"\nFinal Decisions:")
        print(df['final_decision'].value_counts())
        
        return df
    
    def evaluate_predictions(self, df: pd.DataFrame, actual_col: str = 'default_label') -> Dict:
        """Evaluate predictions against actual outcomes"""
        
        if actual_col not in df.columns:
            print(f"Warning: No actual labels found in column '{actual_col}'")
            return {}
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix, classification_report
        )
        
        y_true = df[actual_col]
        y_pred = df['ml_prediction']
        y_proba = df['default_probability']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        print("\nModel Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
        print(f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Repay', 'Default']))
        
        return metrics
    
    def explain_decision(self, farmer_dict: Dict) -> Dict:
        """Provide detailed explanation of credit decision"""
        
        result = self.predict_single(farmer_dict)
        
        # Analyze key factors
        explanation = {
            'farmer_id': result['farmer_id'],
            'final_decision': result['decisions']['final'],
            'credit_score': result['credit_scores']['final_score'],
            'default_probability': result['ml_prediction']['default_probability'],
            'key_strengths': [],
            'key_weaknesses': [],
            'recommendations': []
        }
        
        # Analyze scores
        scores = result['credit_scores']
        
        # Strengths
        if scores['agro_score'] >= 70:
            explanation['key_strengths'].append(
                f"Strong agricultural conditions (score: {scores['agro_score']})"
            )
        if scores['market_score'] >= 70:
            explanation['key_strengths'].append(
                f"Favorable market conditions (score: {scores['market_score']})"
            )
        if scores['repayment_score'] >= 70:
            explanation['key_strengths'].append(
                f"Good repayment capacity (score: {scores['repayment_score']})"
            )
        
        # Weaknesses
        if scores['agro_score'] < 50:
            explanation['key_weaknesses'].append(
                f"Poor agricultural conditions (score: {scores['agro_score']})"
            )
            explanation['recommendations'].append(
                "Consider irrigation or different crop selection"
            )
        if scores['market_score'] < 50:
            explanation['key_weaknesses'].append(
                f"Unfavorable market conditions (score: {scores['market_score']})"
            )
            explanation['recommendations'].append(
                "Join cooperative for better market access"
            )
        if scores['repayment_score'] < 50:
            explanation['key_weaknesses'].append(
                f"Low repayment capacity (score: {scores['repayment_score']})"
            )
            explanation['recommendations'].append(
                "Consider smaller loan amount or longer repayment period"
            )
        
        # Default probability analysis
        if result['ml_prediction']['default_probability'] > 0.5:
            explanation['key_weaknesses'].append(
                f"High default risk: {result['ml_prediction']['default_probability']:.1%}"
            )
        
        return explanation


if __name__ == "__main__":
    # Test prediction
    predictor = CreditPredictor()
    
    # Check if model exists
    model_path = os.path.join(MODELS_DIR, TRAINED_MODEL_FILE)
    if not os.path.exists(model_path):
        print("No trained model found. Please run model_trainer.py first.")
    else:
        # Load test data
        data_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
        if os.path.exists(data_path):
            print("Loading test data...")
            df = pd.read_csv(data_path)
            
            # Take a sample for testing
            test_df = df.sample(n=min(50, len(df)), random_state=RANDOM_SEED)
            
            print("\nMaking predictions...")
            predictions = predictor.predict_with_scores(test_df)
            
            print("\nEvaluating predictions...")
            metrics = predictor.evaluate_predictions(predictions)
            
            # Save predictions
            output_path = f"{RESULTS_DIR}/test_predictions.csv"
            os.makedirs(RESULTS_DIR, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to {output_path}")
            
            # Test single prediction
            print("\nTesting single prediction...")
            sample = predictions.iloc[0].to_dict()
            explanation = predictor.explain_decision(sample)
            
            print("\nDecision Explanation:")
            print(f"Decision: {explanation['final_decision']}")
            print(f"Credit Score: {explanation['credit_score']}")
            print(f"Default Probability: {explanation['default_probability']:.1%}")
            print(f"\nStrengths: {', '.join(explanation['key_strengths']) if explanation['key_strengths'] else 'None'}")
            print(f"Weaknesses: {', '.join(explanation['key_weaknesses']) if explanation['key_weaknesses'] else 'None'}")
            print(f"Recommendations: {', '.join(explanation['recommendations']) if explanation['recommendations'] else 'None'}")
        else:
            print(f"No data found at {data_path}")
            print("Please run data_generator.py first.")
