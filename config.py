"""
Configuration settings for the Farmer Credit Scoring System
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data generation settings
SYNTHETIC_SAMPLE_SIZE = 500

# Model settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Credit scoring thresholds
APPROVE_THRESHOLD = 70
MANUAL_REVIEW_MIN = 50
MANUAL_REVIEW_MAX = 69

# Score weights
AGRO_WEIGHT = 0.30
MARKET_WEIGHT = 0.25
REPAYMENT_WEIGHT = 0.45

# Agro score component weights
CLIMATE_MATCH_WEIGHT = 0.35
SOIL_SUITABILITY_WEIGHT = 0.30
YIELD_PERFORMANCE_WEIGHT = 0.25
IRRIGATION_WEIGHT = 0.10

# Market score component weights
PRICE_TREND_WEIGHT = 0.40
HARVEST_PRICE_WEIGHT = 0.35
VOLATILITY_WEIGHT = 0.15
MARKET_ACCESS_WEIGHT = 0.10

# Repayment score component weights
INCOME_COVERAGE_WEIGHT = 0.40
TRANSACTION_HEALTH_WEIGHT = 0.35
SAVINGS_PATTERN_WEIGHT = 0.15
SOCIAL_CAPITAL_WEIGHT = 0.10

# Default rate and risk factors
BASE_DEFAULT_RATE = 0.20
DROUGHT_RISK_THRESHOLD = 0.7
DROUGHT_DEFAULT_INCREASE = 0.30
HIGH_LTI_THRESHOLD = 0.8
HIGH_LTI_DEFAULT_INCREASE = 0.40
NO_COOP_DEFAULT_INCREASE = 0.15
LOW_MPESA_DEFAULT_INCREASE = 0.25
PRICE_CRASH_DEFAULT_INCREASE = 0.35

# Agro-ecological zones
AGRO_ZONES = {
    'LH1': {'name': 'High Potential Tea Zone', 'min_rainfall': 1500, 'suitable_crops': ['Tea', 'Coffee', 'Dairy']},
    'UM2': {'name': 'Medium-High Maize Zone', 'min_rainfall': 1000, 'suitable_crops': ['Maize', 'Beans', 'Potatoes']},
    'UM3': {'name': 'Medium Maize-Sorghum Zone', 'min_rainfall': 800, 'suitable_crops': ['Maize', 'Sorghum']},
    'LM4': {'name': 'Semi-Arid Zone', 'min_rainfall': 600, 'suitable_crops': ['Millet', 'Sorghum']},
    'L5': {'name': 'Arid Livestock Zone', 'min_rainfall': 300, 'suitable_crops': ['Millet', 'Livestock']}
}

# Crop types distribution
CROP_DISTRIBUTION = {
    'Maize': 0.40,
    'Beans': 0.20,
    'Potatoes': 0.15,
    'Vegetables': 0.15,
    'Other': 0.10
}

# File paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
SYNTHETIC_DATA_FILE = 'synthetic_farmers.csv'
TRAINED_MODEL_FILE = 'credit_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
FEATURE_IMPORTANCE_FILE = 'feature_importance.csv'
