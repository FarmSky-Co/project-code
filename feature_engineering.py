"""
Feature engineering module for credit scoring
Implements all 15 derived features as specified
"""

import pandas as pd
import numpy as np
from typing import Dict


class FeatureEngineer:
    """Create derived features from raw farmer data"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Financial Features (1-5):
        1. Debt-to-Income Ratio
        2. Transaction Velocity
        3. Savings Ratio
        4. Harvest Peak Index
        5. Liquidity Score
        """
        # Calculate expected harvest income
        df['expected_harvest_income'] = (
            df['historical_yield'] * 
            df['farm_size'] * 
            df['projected_harvest_price'] * 1000  # Convert to KES
        )
        
        # 1. Debt-to-Income Ratio
        df['debt_to_income_ratio'] = df['loan_amount_kes'] / df['expected_harvest_income'].replace(0, 1)
        
        # 2. Transaction Velocity (transactions per month)
        df['transaction_velocity'] = df['avg_monthly_transactions'] / df['account_age_months'].replace(0, 1)
        
        # 3. Savings Ratio
        df['savings_ratio'] = df['savings_rate']  # Already calculated in data generation
        
        # 4. Harvest Peak Index - need to parse peak months
        # Simplified: count of peak months / 12
        df['harvest_peak_index'] = df['peak_transaction_months'].str.split(',').str.len() / 12
        
        # 5. Liquidity Score - based on deposits vs withdrawals
        total_flow = df['total_deposits_6m'] + df['total_withdrawals_6m']
        df['liquidity_score'] = (
            df['total_deposits_6m'] / total_flow.replace(0, 1)
        )
        
        return df
    
    def create_agricultural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agricultural Features (6-10):
        6. Yield Efficiency
        7. Climate Match Score
        8. Input Investment Ratio
        9. Experience Factor
        10. Diversification Index
        """
        # Calculate agro zone average yields (simplified)
        zone_avg_yields = df.groupby('agro_zone')['historical_yield'].transform('mean')
        
        # 6. Yield Efficiency
        df['yield_efficiency'] = df['historical_yield'] / zone_avg_yields.replace(0, 1)
        
        # 7. Climate Match Score (combined temperature + rainfall suitability)
        df['climate_match_score'] = (
            df['climate_suitability'] * 0.6 + 
            (1 - df['drought_index']) * 0.4
        )
        
        # 8. Input Investment Ratio (loan per acre)
        df['input_investment_ratio'] = df['loan_amount_kes'] / df['farm_size']
        
        # 9. Experience Factor
        education_multiplier = {
            'None': 0.7, 
            'Primary': 0.85, 
            'Secondary': 1.0, 
            'Tertiary': 1.2
        }
        df['education_multiplier'] = df['education_level'].map(education_multiplier)
        df['experience_factor'] = df['years_farming'] * df['education_multiplier']
        
        # 10. Diversification Index
        # Cooperative membership + group farming + extension access
        df['diversification_index'] = (
            df['cooperative_member'].astype(int) + 
            df['group_farming'].astype(int) + 
            df['extension_access'].astype(int)
        ) / 3
        
        return df
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Risk Features (11-15):
        11. Weather Risk Score
        12. Market Risk Score
        13. Repayment Capacity
        14. Social Capital Score
        15. Default Probability (Target)
        """
        # 11. Weather Risk Score
        df['weather_risk_score'] = (
            df['rainfall_variability'] * df['drought_index']
        )
        
        # 12. Market Risk Score
        # Higher volatility and falling prices = higher risk
        price_trend_score = df['price_trend'].map({
            'Rising': 0.2,
            'Stable': 0.5,
            'Falling': 1.0
        })
        normalized_volatility = df['price_volatility'] / df['current_price_kes']
        df['market_risk_score'] = (
            normalized_volatility * 0.5 + price_trend_score * 0.5
        )
        
        # 13. Repayment Capacity
        # Expected income / (loan + 10% interest)
        loan_with_interest = df['loan_amount_kes'] * 1.10
        df['repayment_capacity'] = (
            df['expected_harvest_income'] / loan_with_interest.replace(0, 1)
        )
        
        # 14. Social Capital Score
        df['social_capital_score'] = (
            df['cooperative_member'].astype(float) * 0.4 +
            (df['training_sessions'] / 20) * 0.3 +  # Normalized to 0-1
            df['group_farming'].astype(float) * 0.2 +
            df['extension_access'].astype(float) * 0.1
        )
        
        # 15. Default Probability is the target variable (already in data)
        # We'll use default_label as our target
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features"""
        print("Creating financial features...")
        df = self.create_financial_features(df)
        
        print("Creating agricultural features...")
        df = self.create_agricultural_features(df)
        
        print("Creating risk features...")
        df = self.create_risk_features(df)
        
        # Store feature names
        self.feature_names = [
            # Financial
            'debt_to_income_ratio', 'transaction_velocity', 'savings_ratio',
            'harvest_peak_index', 'liquidity_score',
            # Agricultural
            'yield_efficiency', 'climate_match_score', 'input_investment_ratio',
            'experience_factor', 'diversification_index',
            # Risk
            'weather_risk_score', 'market_risk_score', 'repayment_capacity',
            'social_capital_score'
        ]
        
        print(f"✓ Created {len(self.feature_names)} engineered features")
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of all engineered feature names"""
        return self.feature_names
    
    def get_base_features(self) -> list:
        """Get important base features to include in modeling"""
        return [
            # Demographics
            'age', 'dependents',
            # Farm characteristics
            'farm_size', 'historical_yield', 'irrigation_access', 'years_farming',
            # Financial
            'account_age_months', 'avg_monthly_transactions', 'avg_transaction_value',
            'total_deposits_6m', 'total_withdrawals_6m', 'transaction_consistency',
            # Agro-ecological
            'rainfall_12m_mm', 'rainfall_variability', 'drought_index',
            'soil_suitability', 'climate_suitability',
            # Market
            'current_price_kes', 'price_volatility',
            # Community
            'cooperative_member', 'training_sessions', 'group_farming', 'extension_access',
            # Loan
            'loan_amount_kes', 'repayment_months'
        ]
    
    def get_all_model_features(self) -> list:
        """Get all features for modeling (base + engineered)"""
        # Categorical encoding
        categorical_features = [
            'gender_encoded',
            'education_encoded',
            'land_ownership_encoded',
            'price_trend_encoded'
        ]
        
        return self.get_base_features() + self.feature_names + categorical_features
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        # Gender
        df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
        
        # Education level
        education_map = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
        df['education_encoded'] = df['education_level'].map(education_map)
        
        # Land ownership
        ownership_map = {'Owned': 2, 'Leased': 1, 'Shared': 0}
        df['land_ownership_encoded'] = df['land_ownership'].map(ownership_map)
        
        # Price trend
        trend_map = {'Rising': 1, 'Stable': 0, 'Falling': -1}
        df['price_trend_encoded'] = df['price_trend'].map(trend_map)
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        
        # Create all features
        df = self.create_all_features(df)
        
        # Encode categorical variables
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        # Handle missing values
        print("Handling missing values...")
        df = df.fillna(0)
        
        # Handle infinities
        df = df.replace([np.inf, -np.inf], 0)
        
        print("✓ Feature engineering complete")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics of engineered features"""
        feature_cols = self.get_all_model_features()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        summary = df[available_cols].describe().T
        summary['missing_pct'] = (df[available_cols].isna().sum() / len(df) * 100)
        
        return summary


if __name__ == "__main__":
    # Test feature engineering
    from data_generator import generate_and_save_data
    
    print("Generating synthetic data...")
    df = generate_and_save_data()
    
    print("\nApplying feature engineering...")
    engineer = FeatureEngineer()
    df_engineered = engineer.prepare_for_modeling(df)
    
    print("\nFeature summary:")
    summary = engineer.get_feature_summary(df_engineered)
    print(summary)
    
    print(f"\nTotal features for modeling: {len(engineer.get_all_model_features())}")
    print("\nFeature names:")
    for i, feat in enumerate(engineer.get_all_model_features(), 1):
        print(f"{i}. {feat}")
