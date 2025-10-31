# 📋 Farmer Credit Scoring System - Complete File Index

## 🎯 Quick Access

**To Get Started:**
1. Read `QUICKSTART.md` (5 minutes)
2. Run `setup.ps1` (automated setup)
3. Launch `streamlit run app.py`

**For Details:**
- Comprehensive guide: `README.md`
- Project overview: `PROJECT_SUMMARY.md`

---

## 📁 Complete File Structure

### 🔧 Setup & Configuration Files
```
📄 setup.ps1                    # Automated setup script (Windows)
📄 requirements.txt             # Python dependencies
📄 config.py                    # System configuration
📄 .gitignore                   # Git ignore rules
```

### 🎨 User Interfaces
```
📄 app.py                       # Streamlit web interface (PRIMARY)
📄 main.py                      # Command-line interface
```

### 📚 Documentation
```
📄 README.md                    # Comprehensive documentation (15+ pages)
📄 QUICKSTART.md                # Quick start guide (2 pages)
📄 PROJECT_SUMMARY.md           # Project overview & features
📄 FILE_INDEX.md                # This file
```

### 🧩 Core Application Modules
```
📄 data_generator.py            # Synthetic data generation
   - SyntheticDataGenerator class
   - Generates 500 farmer records
   - 7 comprehensive data tables
   - Realistic distributions

📄 feature_engineering.py       # Feature engineering
   - FeatureEngineer class
   - 15 derived features
   - Categorical encoding
   - Data preprocessing

📄 scoring_engine.py            # Rule-based credit scoring
   - CreditScoringEngine class
   - Agro-Ecological Score (30%)
   - Market Risk Score (25%)
   - Repayment Score (45%)

📄 model_trainer.py             # ML model training
   - CreditModelTrainer class
   - 5 ML algorithms
   - SMOTE balancing
   - Model comparison
   - Feature importance

📄 predictor.py                 # Prediction & evaluation
   - CreditPredictor class
   - Batch predictions
   - Single predictions
   - Decision explanations
   - Model evaluation
```

### 📊 Data Models Package
```
📁 models/
   📄 __init__.py               # Package initialization
   📄 data_models.py            # Pydantic schemas
      - FarmerDemographics
      - FarmCharacteristics
      - FinancialBehavior
      - AgroEcologicalData
      - MarketIndicators
      - CommunityCooperative
      - LoanApplication
      - CompleteFarmerRecord
      - CreditScoreComponents
```

### 📂 Runtime Directories
```
📁 data/                        # Generated datasets (runtime)
   📄 .gitkeep                  # Keep directory in git
   📄 synthetic_farmers.csv     # Generated training data

📁 models/                      # Trained models (runtime)
   📄 credit_model.pkl          # Best trained model
   📄 feature_scaler.pkl        # Feature scaler
   📄 feature_names.pkl         # Feature list
   📄 feature_importance.csv    # Feature rankings

📁 results/                     # Predictions (runtime)
   📄 .gitkeep                  # Keep directory in git
   📄 test_predictions.csv      # Sample predictions
```

---

## 🎯 File Purposes

### For Installation & Setup
1. **setup.ps1** - Run this first for automated setup
2. **requirements.txt** - Lists all Python packages needed
3. **config.py** - Customize thresholds, weights, paths

### For Usage
**Recommended:**
- **app.py** - Web interface (easiest to use)

**Alternative:**
- **main.py** - Command-line interface
- Individual scripts: `data_generator.py`, `model_trainer.py`, etc.

### For Understanding
1. **QUICKSTART.md** - Start here (5 min read)
2. **PROJECT_SUMMARY.md** - Overview of features
3. **README.md** - Complete documentation
4. **FILE_INDEX.md** - This file

### For Development
- **models/data_models.py** - Data schema definitions
- **feature_engineering.py** - Add custom features
- **config.py** - Adjust scoring parameters
- **scoring_engine.py** - Modify scoring logic

---

## 📖 Module Descriptions

### 1. data_generator.py (450 lines)
**Purpose:** Generate synthetic farmer data for training

**Key Functions:**
- `generate_demographics()` - Age, gender, education, location
- `generate_farm_characteristics()` - Farm size, crops, yields
- `generate_financial_behavior()` - M-Pesa transactions
- `generate_agro_ecological_data()` - Climate, soil, weather
- `generate_market_indicators()` - Prices, trends
- `generate_community_cooperative()` - Social capital
- `generate_loan_applications()` - Loan details
- `generate_default_labels()` - Ground truth labels

**Usage:**
```python
from data_generator import generate_and_save_data
df = generate_and_save_data()
```

### 2. feature_engineering.py (280 lines)
**Purpose:** Create derived features from raw data

**Key Features Created:**
- Financial: Debt-to-income, transaction velocity, savings
- Agricultural: Yield efficiency, climate match, experience
- Risk: Weather risk, market risk, repayment capacity

**Usage:**
```python
from feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
df = engineer.prepare_for_modeling(df)
```

### 3. scoring_engine.py (350 lines)
**Purpose:** Calculate rule-based credit scores

**Components:**
- Agro Score (climate, soil, yield, irrigation)
- Market Score (prices, trends, volatility, access)
- Repayment Score (income, transactions, savings, social)
- Final Score (weighted combination)
- Decision (Approve/Manual Review/Decline)

**Usage:**
```python
from scoring_engine import CreditScoringEngine
scorer = CreditScoringEngine()
df_scored = scorer.score_applications(df)
```

### 4. model_trainer.py (420 lines)
**Purpose:** Train and evaluate ML models

**Supported Models:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Features:**
- SMOTE for class balancing
- Automatic model selection
- Feature importance analysis
- Model persistence

**Usage:**
```python
from model_trainer import CreditModelTrainer
trainer = CreditModelTrainer()
results = trainer.train_all_models(X_train, X_test, y_train, y_test)
trainer.save_model()
```

### 5. predictor.py (320 lines)
**Purpose:** Make predictions on new applications

**Capabilities:**
- Load trained models
- Batch predictions
- Single predictions
- Decision explanations
- Model evaluation

**Usage:**
```python
from predictor import CreditPredictor
predictor = CreditPredictor()
predictions = predictor.predict_with_scores(df)
```

### 6. app.py (650 lines)
**Purpose:** Interactive web interface

**Pages:**
1. Home - Overview and status
2. Data Generation - Create training data
3. Train Model - Train and compare models
4. Make Predictions - Batch evaluation
5. Analytics Dashboard - Insights and charts
6. Single Application - Individual assessment

**Usage:**
```powershell
streamlit run app.py
```

### 7. main.py (180 lines)
**Purpose:** Command-line interface

**Commands:**
- `generate` - Create synthetic data
- `train` - Train ML models
- `predict` - Make predictions
- `score` - Calculate credit scores
- `web` - Launch web interface

**Usage:**
```powershell
python main.py --help
python main.py generate --sample-size 1000
python main.py train --models "XGBoost" "LightGBM"
```

---

## 📊 Data Files (Generated at Runtime)

### data/synthetic_farmers.csv
- 500 rows (configurable)
- 60+ columns
- Complete farmer records
- Ground truth labels
- ~500 KB file size

### models/credit_model.pkl
- Trained ML model (best performing)
- Typically XGBoost or LightGBM
- ~2-5 MB file size

### models/feature_scaler.pkl
- StandardScaler fitted on training data
- Required for predictions
- ~50 KB file size

---

## 🔍 Key Code Locations

### To modify scoring weights:
**File:** `config.py`
**Lines:** 18-36
```python
AGRO_WEIGHT = 0.30
MARKET_WEIGHT = 0.25
REPAYMENT_WEIGHT = 0.45
```

### To add new features:
**File:** `feature_engineering.py`
**Function:** `create_all_features()`
**Lines:** 140-160

### To change decision thresholds:
**File:** `config.py`
**Lines:** 15-17
```python
APPROVE_THRESHOLD = 70
MANUAL_REVIEW_MIN = 50
```

### To add new ML models:
**File:** `model_trainer.py`
**Function:** `train_all_models()`
**Lines:** 200-250

### To customize data generation:
**File:** `data_generator.py`
**Class:** `SyntheticDataGenerator`
**Lines:** 30-450

---

## 🎓 Learning Path

**Beginner:**
1. Read QUICKSTART.md
2. Run setup.ps1
3. Use app.py web interface
4. Generate data → Train → Predict

**Intermediate:**
1. Read README.md
2. Use main.py CLI
3. Modify config.py parameters
4. Review generated data

**Advanced:**
1. Study module code
2. Add custom features
3. Implement new models
4. Extend web interface
5. Integrate real data sources

---

## 🚀 Quick Commands Reference

### Setup
```powershell
# Automated setup
.\setup.ps1

# Manual setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Application
```powershell
# Web interface (recommended)
streamlit run app.py

# Command line
python main.py web
```

### Generate & Train
```powershell
# All-in-one
python data_generator.py
python model_trainer.py

# OR using CLI
python main.py generate
python main.py train
```

### Make Predictions
```powershell
# Using scripts
python predictor.py

# Using CLI
python main.py predict --input data/farmers.csv
```

---

## 📦 Total Project Size

- **Source Code:** ~3,500 lines
- **Documentation:** ~2,000 lines
- **Total Files:** 20+ files
- **Package Size:** ~50 MB (with dependencies)
- **Runtime Data:** ~10 MB (generated)

---

## ✅ Verification Checklist

Before using the system, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Directories created (data/, models/, results/)
- [ ] Can import modules (`python -c "import streamlit"`)
- [ ] Web app launches (`streamlit run app.py`)

---

## 🎯 Next Steps

1. **First Time:**
   - Run `setup.ps1`
   - Read `QUICKSTART.md`
   - Launch `streamlit run app.py`

2. **Learning:**
   - Read `PROJECT_SUMMARY.md`
   - Review `README.md`
   - Explore code modules

3. **Using:**
   - Generate data
   - Train models
   - Make predictions
   - Analyze results

4. **Customizing:**
   - Modify `config.py`
   - Add features in `feature_engineering.py`
   - Adjust scoring in `scoring_engine.py`

---

**Last Updated:** October 30, 2025  
**Version:** 1.0.0  
**Status:** Production Ready ✅
