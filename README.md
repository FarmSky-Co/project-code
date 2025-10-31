# 🌾 Farmer Credit Scoring System

A comprehensive machine learning system for evaluating credit applications from farmers based on agricultural, market, and financial risk factors.

## Overview

This system helps financial institutions make informed lending decisions for farmers using a three-tier scoring framework:

- **Agro-Ecological Risk Score (30%)**: Climate, soil, yield, irrigation
- **Market Risk Score (25%)**: Price trends, volatility, market access
- **Repayment Risk Score (45%)**: Income coverage, M-Pesa transactions, savings, social capital

**Final Score Formula:**
```
Final Score = (Agro Score × 0.30) + (Market Score × 0.25) + (Repayment Score × 0.45)
```

**Decision Thresholds:**
- ✅ Approve: Score ≥ 70
- ⚠️ Manual Review: Score 50-69
- ❌ Decline: Score < 50

## Features

✨ **Complete Credit Scoring Framework**
- Rule-based scoring using agricultural and financial metrics
- Machine learning models for default prediction
- Hybrid decision-making combining both approaches

📊 **Advanced Analytics**
- 15 engineered features (financial, agricultural, risk)
- Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Comprehensive model evaluation and comparison

🎨 **Interactive Web Interface**
- User-friendly Streamlit dashboard
- Data generation and visualization
- Model training interface
- Batch and single application predictions
- Analytics dashboard with insights

🔧 **Production-Ready**
- Modular architecture
- Model persistence and versioning
- Comprehensive error handling
- Extensive documentation

## Project Structure

```
farmer-credit-scoring/
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Data models
│   ├── __init__.py
│   └── data_models.py          # Pydantic schemas
│
├── data_generator.py           # Synthetic data generation
├── feature_engineering.py      # Feature creation
├── scoring_engine.py           # Rule-based scoring
├── model_trainer.py            # ML model training
├── predictor.py                # Prediction engine
│
├── data/                       # Generated data (created at runtime)
│   └── synthetic_farmers.csv
│
├── models/                     # Trained models (created at runtime)
│   ├── credit_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_importance.csv
│
└── results/                    # Prediction results (created at runtime)
    └── test_predictions.csv
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

Download the project to your local machine.

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an error, you may need to run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install all required packages including:
- pandas, numpy, scikit-learn (data processing & ML)
- xgboost, lightgbm (advanced ML algorithms)
- streamlit (web interface)
- plotly (visualizations)
- pydantic (data validation)
- faker (synthetic data generation)

## Quick Start

### Method 1: Using the Web Interface (Recommended)

1. **Start the application:**
   ```powershell
   streamlit run app.py
   ```

2. **Generate Training Data:**
   - Navigate to "Data Generation" page
   - Click "Generate Data" button
   - Wait for 500 farmer records to be created

3. **Train Model:**
   - Navigate to "Train Model" page
   - Select models to train (e.g., Random Forest, XGBoost, LightGBM)
   - Click "Start Training"
   - Best model will be automatically saved

4. **Make Predictions:**
   - Navigate to "Make Predictions" for batch predictions
   - Or "Single Application" for individual evaluation
   - View results and download predictions

5. **View Analytics:**
   - Navigate to "Analytics Dashboard"
   - Explore score distributions, geographic analysis, and crop insights

### Method 2: Using Command Line

1. **Generate synthetic data:**
   ```powershell
   python data_generator.py
   ```

2. **Train models:**
   ```powershell
   python model_trainer.py
   ```

3. **Make predictions:**
   ```powershell
   python predictor.py
   ```

4. **Calculate credit scores:**
   ```powershell
   python scoring_engine.py
   ```

## Usage Examples

### Example 1: Generate Custom Data

```python
from data_generator import SyntheticDataGenerator

# Generate 1000 farmer records
generator = SyntheticDataGenerator(sample_size=1000)
df = generator.generate_complete_dataset()
df.to_csv('data/my_farmers.csv', index=False)
```

### Example 2: Train Custom Model

```python
from model_trainer import CreditModelTrainer
from feature_engineering import FeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('data/synthetic_farmers.csv')

# Engineer features
engineer = FeatureEngineer()
df = engineer.prepare_for_modeling(df)

# Train model
trainer = CreditModelTrainer()
feature_cols = engineer.get_all_model_features()
X_train, X_test, y_train, y_test = trainer.prepare_data(df, feature_cols)
results = trainer.train_all_models(X_train, X_test, y_train, y_test)

# Save best model
trainer.save_model()
```

### Example 3: Score Single Application

```python
from predictor import CreditPredictor

predictor = CreditPredictor()

farmer = {
    'farmer_id': 'FRM00001',
    'age': 42,
    'farm_size': 3.5,
    'crop_type': 'Maize',
    'loan_amount_kes': 50000,
    # ... other fields
}

result = predictor.predict_single(farmer)
print(f"Decision: {result['decisions']['final']}")
print(f"Credit Score: {result['credit_scores']['final_score']}")
print(f"Default Probability: {result['ml_prediction']['default_probability']}")
```

### Example 4: Batch Predictions

```python
from predictor import CreditPredictor

predictor = CreditPredictor()
predictions = predictor.batch_predict(
    input_file='data/new_applications.csv',
    output_file='results/predictions.csv'
)

# Get summary
print(predictions['final_decision'].value_counts())
```

## System Components

### 1. Data Generation (`data_generator.py`)

Generates realistic synthetic farmer data with:
- Demographics (age, gender, education, location)
- Farm characteristics (size, crops, yields, irrigation)
- Financial behavior (M-Pesa transactions, savings)
- Agro-ecological data (climate, soil, weather)
- Market indicators (prices, trends, volatility)
- Loan applications
- Ground truth default labels

### 2. Feature Engineering (`feature_engineering.py`)

Creates 15 derived features:

**Financial Features:**
1. Debt-to-Income Ratio
2. Transaction Velocity
3. Savings Ratio
4. Harvest Peak Index
5. Liquidity Score

**Agricultural Features:**
6. Yield Efficiency
7. Climate Match Score
8. Input Investment Ratio
9. Experience Factor
10. Diversification Index

**Risk Features:**
11. Weather Risk Score
12. Market Risk Score
13. Repayment Capacity
14. Social Capital Score
15. Default Probability (Target)

### 3. Scoring Engine (`scoring_engine.py`)

Rule-based credit scoring with three components:

**Agro-Ecological Score (30%):**
- Climate suitability (35%)
- Soil quality (30%)
- Yield performance (25%)
- Irrigation (10%)

**Market Score (25%):**
- Price trends (40%)
- Harvest pricing (35%)
- Volatility (15%)
- Market access (10%)

**Repayment Score (45%):**
- Income coverage (40%)
- Transaction health (35%)
- Savings pattern (15%)
- Social capital (10%)

### 4. Model Training (`model_trainer.py`)

Supports multiple ML algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Features:
- SMOTE for class balancing
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Model persistence

### 5. Prediction Engine (`predictor.py`)

Makes predictions using:
- Trained ML models
- Rule-based credit scores
- Hybrid decision-making
- Batch and single predictions
- Decision explanations

### 6. Web Interface (`app.py`)

Streamlit dashboard with:
- Home page with system overview
- Data generation interface
- Model training interface
- Batch prediction tool
- Single application evaluator
- Analytics dashboard

## Configuration

Edit `config.py` to customize:

```python
# Sample size
SYNTHETIC_SAMPLE_SIZE = 500

# Model settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Score thresholds
APPROVE_THRESHOLD = 70
MANUAL_REVIEW_MIN = 50

# Score weights
AGRO_WEIGHT = 0.30
MARKET_WEIGHT = 0.25
REPAYMENT_WEIGHT = 0.45

# Default risk factors
BASE_DEFAULT_RATE = 0.20
DROUGHT_RISK_THRESHOLD = 0.7
HIGH_LTI_THRESHOLD = 0.8
```

## Data Schema

### Farmer Demographics
- farmer_id, name, gender, age
- education_level, dependents
- county, subcounty, coordinates
- id_number, phone_number

### Farm Characteristics
- farm_size, crop_type, historical_yield
- irrigation_access, land_ownership
- years_farming, agro_zone

### Financial Behavior
- account_age_months, avg_monthly_transactions
- total_deposits_6m, total_withdrawals_6m
- savings_rate, transaction_consistency
- peak_transaction_months

### Agro-Ecological Data
- rainfall_12m_mm, rainfall_variability
- drought_index, soil_type, soil_suitability
- temperature_min, temperature_max
- climate_suitability

### Market Indicators
- current_price_kes, price_6m_avg
- price_volatility, projected_harvest_price
- price_trend, expected_harvest_month

### Community & Cooperative
- cooperative_member, cooperative_name
- training_sessions, group_farming
- extension_access, years_in_coop

### Loan Application
- application_id, loan_amount_kes
- loan_purpose, repayment_months
- planting_date, harvest_date

## Model Performance

Expected performance metrics (may vary):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.75 | 0.68 | 0.72 | 0.70 | 0.82 |
| Random Forest | 0.82 | 0.78 | 0.80 | 0.79 | 0.88 |
| Gradient Boosting | 0.81 | 0.77 | 0.79 | 0.78 | 0.87 |
| XGBoost | 0.83 | 0.80 | 0.81 | 0.80 | 0.89 |
| LightGBM | 0.83 | 0.79 | 0.82 | 0.80 | 0.89 |

## Troubleshooting

### Issue: Module not found
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Streamlit won't start
```powershell
# Check if streamlit is installed
pip show streamlit

# If not, install it
pip install streamlit

# Try running with full path
python -m streamlit run app.py
```

### Issue: No trained model found
```powershell
# Train a model first
python model_trainer.py

# Or use the web interface: Data Generation → Train Model
```

### Issue: Memory errors with large datasets
- Reduce `SYNTHETIC_SAMPLE_SIZE` in `config.py`
- Use fewer models during training
- Process data in smaller batches

## Advanced Usage

### Custom Scoring Weights

Modify scoring weights in `config.py`:

```python
# Emphasize repayment capacity
AGRO_WEIGHT = 0.20
MARKET_WEIGHT = 0.20
REPAYMENT_WEIGHT = 0.60
```

### Adding New Features

1. Edit `feature_engineering.py`
2. Add feature creation logic
3. Update feature names list
4. Retrain models

### Custom ML Models

Add new models in `model_trainer.py`:

```python
def train_custom_model(self, X_train, y_train):
    from sklearn.svm import SVC
    
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    
    self.models['Custom SVM'] = model
    return model
```

### Deploying to Production

1. **Set up environment:**
   ```powershell
   pip install gunicorn
   ```

2. **Run with production server:**
   ```powershell
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

3. **Use environment variables:**
   Create `.env` file for sensitive settings

## Contributing

This is a demonstration system. For production use:
- Add authentication and user management
- Implement database integration
- Add API endpoints
- Enhance security measures
- Add comprehensive logging
- Implement model monitoring

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review code comments and docstrings
3. Examine the example usage

## Acknowledgments

Built using:
- scikit-learn for ML models
- XGBoost and LightGBM for gradient boosting
- Streamlit for web interface
- Plotly for visualizations
- Pydantic for data validation
- Faker for synthetic data generation

---

**Author:** Farmer Credit Scoring System Team  
**Version:** 1.0.0  
**Last Updated:** October 2025
#   p r o j e c t - c o d e  
 