"""
Synthetic data generation for farmer credit scoring
Generates realistic farmer data following specified distributions and patterns
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from typing import List, Dict
import random
from config import *

fake = Faker()
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)


class SyntheticDataGenerator:
    """Generate synthetic farmer data"""
    
    def __init__(self, sample_size: int = SYNTHETIC_SAMPLE_SIZE):
        self.sample_size = sample_size
        self.counties = [
            'Nakuru', 'Uasin Gishu', 'Trans Nzoia', 'Narok', 'Bomet',
            'Kericho', 'Nyandarua', 'Kiambu', 'Meru', 'Embu',
            'Machakos', 'Makueni', 'Kitui', 'Bungoma', 'Kakamega'
        ]
        self.soil_types = [
            'Sandy loam', 'Clay loam', 'Silt loam', 'Loamy sand',
            'Sandy clay', 'Silty clay', 'Red volcanic'
        ]
        
    def generate_demographics(self) -> pd.DataFrame:
        """Generate farmer demographics"""
        data = []
        
        for i in range(self.sample_size):
            farmer_id = f"FRM{str(i+1).zfill(5)}"
            gender = np.random.choice(['M', 'F'], p=[0.6, 0.4])
            age = int(np.clip(np.random.normal(42, 12), 22, 70))
            education = np.random.choice(
                ['None', 'Primary', 'Secondary', 'Tertiary'],
                p=[0.15, 0.45, 0.30, 0.10]
            )
            dependents = min(int(np.random.poisson(4)), 12)
            county = np.random.choice(self.counties)
            
            data.append({
                'farmer_id': farmer_id,
                'name': fake.name(),
                'gender': gender,
                'age': age,
                'education_level': education,
                'dependents': dependents,
                'county': county,
                'subcounty': f"{county} {random.choice(['North', 'South', 'East', 'West'])}",
                'latitude': round(random.uniform(-4.5, 4.5), 6),
                'longitude': round(random.uniform(33.5, 41.5), 6),
                'id_number': str(random.randint(10000000, 99999999)),
                'phone_number': f"+254{random.randint(700000000, 799999999)}"
            })
        
        return pd.DataFrame(data)
    
    def generate_farm_characteristics(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate farm characteristics correlated with demographics"""
        data = []
        
        for _, farmer in demographics.iterrows():
            # Farm size: log-normal distribution
            farm_size = round(np.random.lognormal(0.7, 0.8), 2)
            farm_size = np.clip(farm_size, 0.25, 20)
            
            # Crop type selection
            crop_type = np.random.choice(
                list(CROP_DISTRIBUTION.keys()),
                p=list(CROP_DISTRIBUTION.values())
            )
            
            # Agro zone assignment based on location
            agro_zones = list(AGRO_ZONES.keys())
            agro_zone = np.random.choice(agro_zones)
            
            # Years farming correlated with age
            max_farming_years = min(farmer['age'] - 18, 50)
            years_farming = random.randint(1, max(max_farming_years, 1))
            
            # Education affects yield
            education_multiplier = {
                'None': 0.7, 'Primary': 0.85, 'Secondary': 1.0, 'Tertiary': 1.2
            }
            base_yield = np.random.uniform(0.5, 3.0)
            historical_yield = round(
                base_yield * education_multiplier[farmer['education_level']], 2
            )
            
            data.append({
                'farmer_id': farmer['farmer_id'],
                'farm_size': farm_size,
                'crop_type': crop_type,
                'historical_yield': historical_yield,
                'irrigation_access': random.random() < 0.3,  # 30% have irrigation
                'land_ownership': np.random.choice(['Owned', 'Leased', 'Shared'], p=[0.7, 0.2, 0.1]),
                'years_farming': years_farming,
                'agro_zone': agro_zone
            })
        
        return pd.DataFrame(data)
    
    def generate_financial_behavior(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate M-Pesa transaction patterns"""
        data = []
        
        for _, farmer in demographics.iterrows():
            account_age = random.randint(6, 120)  # 6 months to 10 years
            
            # Transaction patterns vary by farmer wealth/education
            base_transactions = 15
            if farmer['education_level'] in ['Secondary', 'Tertiary']:
                base_transactions = 25
            
            avg_monthly_transactions = random.randint(
                int(base_transactions * 0.5), 
                int(base_transactions * 1.5)
            )
            
            avg_transaction_value = round(random.uniform(500, 5000), 2)
            
            # Generate deposits and withdrawals
            total_deposits_6m = round(random.uniform(10000, 150000), 2)
            total_withdrawals_6m = round(
                total_deposits_6m * random.uniform(0.7, 1.1), 2
            )
            
            savings_rate = min(
                round(total_deposits_6m / (total_deposits_6m + total_withdrawals_6m), 3),
                1.0
            )
            
            # Peak transaction months (harvest season)
            harvest_months = random.sample(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                k=random.randint(2, 4)
            )
            
            transaction_consistency = round(random.uniform(0.3, 0.95), 3)
            
            data.append({
                'farmer_id': farmer['farmer_id'],
                'account_age_months': account_age,
                'avg_monthly_transactions': avg_monthly_transactions,
                'avg_transaction_value': avg_transaction_value,
                'total_deposits_6m': total_deposits_6m,
                'total_withdrawals_6m': total_withdrawals_6m,
                'savings_rate': savings_rate,
                'peak_transaction_months': ','.join(harvest_months),
                'transaction_consistency': transaction_consistency
            })
        
        return pd.DataFrame(data)
    
    def generate_agro_ecological_data(self, farm_chars: pd.DataFrame) -> pd.DataFrame:
        """Generate environmental and climate data"""
        data = []
        
        for _, farm in farm_chars.iterrows():
            agro_zone_info = AGRO_ZONES[farm['agro_zone']]
            min_rainfall = agro_zone_info['min_rainfall']
            
            # Rainfall based on agro zone
            rainfall_12m = round(
                random.uniform(min_rainfall, min_rainfall + 500), 2
            )
            rainfall_variability = round(random.uniform(0.1, 0.4), 3)
            
            # Drought index (higher = worse)
            drought_index = round(random.uniform(0, 0.8), 3)
            
            # Soil type and suitability
            soil_type = random.choice(self.soil_types)
            soil_suitability = round(random.uniform(0.4, 1.0), 3)
            
            # Temperature range
            temp_min = round(random.uniform(10, 18), 1)
            temp_max = round(random.uniform(22, 32), 1)
            
            # Climate suitability for crop
            climate_suitability = round(random.uniform(0.5, 1.0), 3)
            
            data.append({
                'farmer_id': farm['farmer_id'],
                'agro_zone_code': farm['agro_zone'],
                'rainfall_12m_mm': rainfall_12m,
                'rainfall_variability': rainfall_variability,
                'drought_index': drought_index,
                'soil_type': soil_type,
                'soil_suitability': soil_suitability,
                'temperature_min': temp_min,
                'temperature_max': temp_max,
                'climate_suitability': climate_suitability
            })
        
        return pd.DataFrame(data)
    
    def generate_market_indicators(self, farm_chars: pd.DataFrame) -> pd.DataFrame:
        """Generate market price data"""
        # Base prices for crops (KES per kg)
        base_prices = {
            'Maize': 45, 'Beans': 80, 'Potatoes': 35,
            'Vegetables': 50, 'Other': 40
        }
        
        data = []
        
        for _, farm in farm_chars.iterrows():
            crop = farm['crop_type']
            base_price = base_prices.get(crop, 45)
            
            current_price = round(base_price * random.uniform(0.8, 1.2), 2)
            price_6m_avg = round(base_price * random.uniform(0.85, 1.15), 2)
            price_volatility = round(base_price * random.uniform(0.05, 0.25), 2)
            
            # Harvest month
            harvest_month = random.choice([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            # Projected harvest price
            projected_harvest_price = round(
                current_price * random.uniform(0.85, 1.15), 2
            )
            
            # Price trend
            if projected_harvest_price > current_price * 1.05:
                price_trend = 'Rising'
            elif projected_harvest_price < current_price * 0.95:
                price_trend = 'Falling'
            else:
                price_trend = 'Stable'
            
            data.append({
                'farmer_id': farm['farmer_id'],
                'crop_type': crop,
                'current_price_kes': current_price,
                'price_6m_avg': price_6m_avg,
                'price_volatility': price_volatility,
                'expected_harvest_month': harvest_month,
                'projected_harvest_price': projected_harvest_price,
                'price_trend': price_trend
            })
        
        return pd.DataFrame(data)
    
    def generate_community_cooperative(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate cooperative and community data"""
        data = []
        
        cooperative_names = [
            'Green Valley Farmers Co-op', 'Highland Agricultural Society',
            'United Farmers Group', 'Progressive Farmers Union',
            'New Hope Cooperative', 'Farmers Pride Association'
        ]
        
        for _, farmer in demographics.iterrows():
            is_member = random.random() < 0.6  # 60% are coop members
            
            data.append({
                'farmer_id': farmer['farmer_id'],
                'cooperative_member': is_member,
                'cooperative_name': random.choice(cooperative_names) if is_member else None,
                'training_sessions': random.randint(0, 20) if is_member else random.randint(0, 5),
                'group_farming': random.random() < 0.4,  # 40% do group farming
                'extension_access': random.random() < 0.5,  # 50% have extension access
                'years_in_coop': random.randint(1, 15) if is_member else None
            })
        
        return pd.DataFrame(data)
    
    def generate_loan_applications(self, farm_chars: pd.DataFrame) -> pd.DataFrame:
        """Generate loan application data"""
        data = []
        
        for idx, farm in farm_chars.iterrows():
            application_id = f"APP{str(idx+1).zfill(5)}"
            
            # Loan amount based on farm size
            loan_amount = round(
                farm['farm_size'] * random.uniform(5000, 15000), 2
            )
            
            loan_purpose = random.choice([
                'Seeds', 'Fertilizer', 'Pesticides', 'Equipment', 'Labor', 'Mixed'
            ])
            
            repayment_months = random.choice([3, 4, 6, 9, 12])
            
            # Generate planting and harvest dates
            application_date = datetime.now() - timedelta(days=random.randint(1, 90))
            planting_date = application_date + timedelta(days=random.randint(7, 30))
            harvest_date = planting_date + timedelta(days=random.randint(90, 180))
            
            data.append({
                'application_id': application_id,
                'farmer_id': farm['farmer_id'],
                'loan_amount_kes': loan_amount,
                'loan_purpose': loan_purpose,
                'repayment_months': repayment_months,
                'planting_date': planting_date.date(),
                'harvest_date': harvest_date.date(),
                'application_date': application_date.date()
            })
        
        return pd.DataFrame(data)
    
    def generate_default_labels(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ground truth default labels based on risk factors"""
        
        def calculate_default_probability(row):
            base_prob = BASE_DEFAULT_RATE
            
            # Drought risk
            if row['drought_index'] > DROUGHT_RISK_THRESHOLD:
                base_prob += DROUGHT_DEFAULT_INCREASE
            
            # Calculate loan-to-income ratio
            expected_income = (
                row['historical_yield'] * 
                row['farm_size'] * 
                row['projected_harvest_price'] * 1000
            )
            lti_ratio = row['loan_amount_kes'] / max(expected_income, 1)
            
            if lti_ratio > HIGH_LTI_THRESHOLD:
                base_prob += HIGH_LTI_DEFAULT_INCREASE
            
            # No cooperative membership
            if not row['cooperative_member']:
                base_prob += NO_COOP_DEFAULT_INCREASE
            
            # Low M-Pesa activity
            if row['avg_monthly_transactions'] < 10:
                base_prob += LOW_MPESA_DEFAULT_INCREASE
            
            # Price crash
            if row['price_trend'] == 'Falling':
                base_prob += PRICE_CRASH_DEFAULT_INCREASE
            
            return min(base_prob, 0.95)  # Cap at 95%
        
        combined_df['default_probability'] = combined_df.apply(
            calculate_default_probability, axis=1
        )
        
        # Generate binary labels
        combined_df['default_label'] = combined_df['default_probability'].apply(
            lambda p: 1 if random.random() < p else 0
        )
        
        return combined_df
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        print("Generating demographics...")
        demographics = self.generate_demographics()
        
        print("Generating farm characteristics...")
        farm_chars = self.generate_farm_characteristics(demographics)
        
        print("Generating financial behavior...")
        financial = self.generate_financial_behavior(demographics)
        
        print("Generating agro-ecological data...")
        agro_eco = self.generate_agro_ecological_data(farm_chars)
        
        print("Generating market indicators...")
        market = self.generate_market_indicators(farm_chars)
        
        print("Generating community data...")
        community = self.generate_community_cooperative(demographics)
        
        print("Generating loan applications...")
        loans = self.generate_loan_applications(farm_chars)
        
        # Merge all datasets
        print("Merging datasets...")
        combined = demographics.merge(farm_chars, on='farmer_id')
        combined = combined.merge(financial, on='farmer_id')
        combined = combined.merge(agro_eco, on='farmer_id')
        combined = combined.merge(market, on='farmer_id')
        combined = combined.merge(community, on='farmer_id')
        combined = combined.merge(loans, on='farmer_id')
        
        print("Generating default labels...")
        combined = self.generate_default_labels(combined)
        
        print(f"✓ Generated {len(combined)} complete farmer records")
        print(f"✓ Default rate: {combined['default_label'].mean():.2%}")
        
        return combined


def generate_and_save_data(output_path: str = None):
    """Generate and save synthetic data to CSV"""
    if output_path is None:
        output_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
    
    generator = SyntheticDataGenerator()
    df = generator.generate_complete_dataset()
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    df = generate_and_save_data()
    print("\nDataset summary:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
