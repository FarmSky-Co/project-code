"""
Credit Scoring Engine
Implements the three-tier scoring system:
- Agro-Ecological Risk Score (30%)
- Market Risk Score (25%)
- Repayment Risk Score (45%)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from config import *


class CreditScoringEngine:
    """Calculate credit scores based on the defined framework"""
    
    def __init__(self):
        self.score_history = []
    
    def calculate_agro_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agro-Ecological Risk Score (30%)
        Components:
        - Climate suitability (35%)
        - Soil quality match (30%)
        - Historical yield performance (25%)
        - Irrigation access (10%)
        """
        # Climate Match (0-100 scale)
        df['climate_match'] = df['climate_suitability'] * 100
        
        # Soil Suitability (0-100 scale)
        df['soil_match'] = df['soil_suitability'] * 100
        
        # Yield Performance - compare to zone average
        zone_avg_yield = df.groupby('agro_zone')['historical_yield'].transform('mean')
        yield_ratio = df['historical_yield'] / zone_avg_yield.replace(0, 1)
        df['yield_performance'] = np.clip(yield_ratio * 50, 0, 100)
        
        # Irrigation Score
        df['irrigation_score'] = df['irrigation_access'].astype(float) * 100
        
        # Calculate weighted Agro Score
        df['agro_score'] = (
            df['climate_match'] * CLIMATE_MATCH_WEIGHT +
            df['soil_match'] * SOIL_SUITABILITY_WEIGHT +
            df['yield_performance'] * YIELD_PERFORMANCE_WEIGHT +
            df['irrigation_score'] * IRRIGATION_WEIGHT
        )
        
        return df
    
    def calculate_market_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market Risk Score (25%)
        Components:
        - Current crop price trends (40%)
        - Expected harvest period pricing (35%)
        - Price volatility index (15%)
        - Market access (10%)
        """
        # Price Trend Score
        price_trend_map = {
            'Rising': 100,
            'Stable': 70,
            'Falling': 30
        }
        df['price_trend_score'] = df['price_trend'].map(price_trend_map)
        
        # Harvest Price Score - compare projected to current
        price_change_ratio = (
            df['projected_harvest_price'] / df['current_price_kes'].replace(0, 1)
        )
        df['harvest_price_score'] = np.clip(price_change_ratio * 70, 20, 100)
        
        # Volatility Score (lower volatility = higher score)
        volatility_ratio = df['price_volatility'] / df['current_price_kes'].replace(0, 1)
        df['volatility_score'] = np.clip(100 - (volatility_ratio * 200), 0, 100)
        
        # Market Access Score - based on cooperative membership and location
        df['market_access_score'] = (
            df['cooperative_member'].astype(float) * 70 +
            30  # Base access score
        )
        
        # Calculate weighted Market Score
        df['market_score'] = (
            df['price_trend_score'] * PRICE_TREND_WEIGHT +
            df['harvest_price_score'] * HARVEST_PRICE_WEIGHT +
            df['volatility_score'] * VOLATILITY_WEIGHT +
            df['market_access_score'] * MARKET_ACCESS_WEIGHT
        )
        
        return df
    
    def calculate_repayment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repayment Risk Score (45%)
        Components:
        - Expected income vs loan amount (40%)
        - M-Pesa transaction patterns (35%)
        - Savings behavior (15%)
        - Cooperative membership (10%)
        """
        # Income Coverage Score
        # Calculate expected income
        expected_income = (
            df['historical_yield'] * 
            df['farm_size'] * 
            df['projected_harvest_price'] * 1000
        )
        
        # Income coverage ratio
        loan_with_interest = df['loan_amount_kes'] * 1.10
        income_coverage_ratio = expected_income / loan_with_interest.replace(0, 1)
        
        # Convert to 0-100 scale
        # Ratio >= 2.0 = 100, Ratio <= 0.5 = 0
        df['income_coverage_score'] = np.clip(
            (income_coverage_ratio - 0.5) / 1.5 * 100, 
            0, 100
        )
        
        # Transaction Health Score
        # Based on transaction consistency and volume
        transaction_score = (
            df['transaction_consistency'] * 50 +
            np.clip(df['avg_monthly_transactions'] / 30 * 50, 0, 50)
        )
        df['transaction_health_score'] = transaction_score
        
        # Savings Pattern Score
        df['savings_pattern_score'] = df['savings_rate'] * 100
        
        # Social Capital Score
        social_capital = (
            df['cooperative_member'].astype(float) * 40 +
            np.clip(df['training_sessions'] / 20 * 30, 0, 30) +
            df['group_farming'].astype(float) * 20 +
            df['extension_access'].astype(float) * 10
        )
        df['social_capital_score'] = social_capital
        
        # Calculate weighted Repayment Score
        df['repayment_score'] = (
            df['income_coverage_score'] * INCOME_COVERAGE_WEIGHT +
            df['transaction_health_score'] * TRANSACTION_HEALTH_WEIGHT +
            df['savings_pattern_score'] * SAVINGS_PATTERN_WEIGHT +
            df['social_capital_score'] * SOCIAL_CAPITAL_WEIGHT
        )
        
        return df
    
    def calculate_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final credit score and decision
        Final Score = (Agro Score × 0.30) + (Market Score × 0.25) + (Repayment Score × 0.45)
        """
        df['final_credit_score'] = (
            df['agro_score'] * AGRO_WEIGHT +
            df['market_score'] * MARKET_WEIGHT +
            df['repayment_score'] * REPAYMENT_WEIGHT
        )
        
        # Make decision based on thresholds
        def make_decision(score):
            if score >= APPROVE_THRESHOLD:
                return 'Approve'
            elif score >= MANUAL_REVIEW_MIN:
                return 'Manual Review'
            else:
                return 'Decline'
        
        df['credit_decision'] = df['final_credit_score'].apply(make_decision)
        
        return df
    
    def score_applications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete scoring pipeline"""
        print("Calculating Agro-Ecological Score...")
        df = self.calculate_agro_score(df)
        
        print("Calculating Market Score...")
        df = self.calculate_market_score(df)
        
        print("Calculating Repayment Score...")
        df = self.calculate_repayment_score(df)
        
        print("Calculating Final Credit Score...")
        df = self.calculate_final_score(df)
        
        print("✓ Credit scoring complete")
        
        return df
    
    def get_score_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of credit scores"""
        summary = {
            'total_applications': len(df),
            'avg_final_score': df['final_credit_score'].mean(),
            'avg_agro_score': df['agro_score'].mean(),
            'avg_market_score': df['market_score'].mean(),
            'avg_repayment_score': df['repayment_score'].mean(),
            'approved': (df['credit_decision'] == 'Approve').sum(),
            'manual_review': (df['credit_decision'] == 'Manual Review').sum(),
            'declined': (df['credit_decision'] == 'Decline').sum(),
            'approval_rate': (df['credit_decision'] == 'Approve').mean() * 100,
            'decline_rate': (df['credit_decision'] == 'Decline').mean() * 100
        }
        
        return summary
    
    def score_single_application(self, farmer_data: Dict) -> Dict:
        """Score a single farmer application"""
        # Convert to DataFrame for consistency
        df = pd.DataFrame([farmer_data])
        
        # Run scoring
        df = self.score_applications(df)
        
        # Extract scores
        result = {
            'farmer_id': df.iloc[0]['farmer_id'],
            'agro_score': round(df.iloc[0]['agro_score'], 2),
            'market_score': round(df.iloc[0]['market_score'], 2),
            'repayment_score': round(df.iloc[0]['repayment_score'], 2),
            'final_score': round(df.iloc[0]['final_credit_score'], 2),
            'decision': df.iloc[0]['credit_decision'],
            'components': {
                'agro': {
                    'climate_match': round(df.iloc[0]['climate_match'], 2),
                    'soil_match': round(df.iloc[0]['soil_match'], 2),
                    'yield_performance': round(df.iloc[0]['yield_performance'], 2),
                    'irrigation_score': round(df.iloc[0]['irrigation_score'], 2)
                },
                'market': {
                    'price_trend_score': round(df.iloc[0]['price_trend_score'], 2),
                    'harvest_price_score': round(df.iloc[0]['harvest_price_score'], 2),
                    'volatility_score': round(df.iloc[0]['volatility_score'], 2),
                    'market_access_score': round(df.iloc[0]['market_access_score'], 2)
                },
                'repayment': {
                    'income_coverage_score': round(df.iloc[0]['income_coverage_score'], 2),
                    'transaction_health_score': round(df.iloc[0]['transaction_health_score'], 2),
                    'savings_pattern_score': round(df.iloc[0]['savings_pattern_score'], 2),
                    'social_capital_score': round(df.iloc[0]['social_capital_score'], 2)
                }
            }
        }
        
        return result
    
    def get_score_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get distribution of scores by decision category"""
        distribution = df.groupby('credit_decision').agg({
            'final_credit_score': ['count', 'mean', 'min', 'max', 'std'],
            'agro_score': 'mean',
            'market_score': 'mean',
            'repayment_score': 'mean'
        }).round(2)
        
        return distribution
    
    def analyze_declined_applications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze why applications were declined"""
        declined = df[df['credit_decision'] == 'Decline'].copy()
        
        if len(declined) == 0:
            return pd.DataFrame()
        
        analysis = {
            'count': len(declined),
            'avg_agro_score': declined['agro_score'].mean(),
            'avg_market_score': declined['market_score'].mean(),
            'avg_repayment_score': declined['repayment_score'].mean(),
            'low_agro_pct': (declined['agro_score'] < 40).mean() * 100,
            'low_market_pct': (declined['market_score'] < 40).mean() * 100,
            'low_repayment_pct': (declined['repayment_score'] < 40).mean() * 100,
            'high_drought_pct': (declined['drought_index'] > 0.6).mean() * 100,
            'no_coop_pct': (~declined['cooperative_member']).mean() * 100
        }
        
        return pd.DataFrame([analysis])


if __name__ == "__main__":
    # Test scoring engine
    from data_generator import generate_and_save_data
    import os
    
    print("Generating synthetic data...")
    if not os.path.exists(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"):
        df = generate_and_save_data()
    else:
        df = pd.read_csv(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}")
    
    print("\nRunning credit scoring...")
    scorer = CreditScoringEngine()
    df_scored = scorer.score_applications(df)
    
    print("\nScore Summary:")
    summary = scorer.get_score_summary(df_scored)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nScore Distribution by Decision:")
    print(scorer.get_score_distribution(df_scored))
    
    print("\nDeclined Applications Analysis:")
    print(scorer.analyze_declined_applications(df_scored))
    
    # Save scored data
    output_path = f"{DATA_DIR}/scored_applications.csv"
    df_scored.to_csv(output_path, index=False)
    print(f"\n✓ Scored data saved to {output_path}")
