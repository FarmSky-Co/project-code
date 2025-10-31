# Farmer Credit Scoring System - Quick Start Guide

## Installation (Windows PowerShell)

1. **Open PowerShell in the project directory**

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # If you get an error, run:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Option 1: Web Interface (Easiest)

```powershell
streamlit run app.py
```

Then follow the steps in the web interface:
1. Go to "Data Generation" → Generate Data
2. Go to "Train Model" → Start Training
3. Go to "Make Predictions" or "Single Application"

## Option 2: Command Line

### Step 1: Generate Data
```powershell
python main.py generate --sample-size 500
```

### Step 2: Train Model
```powershell
python main.py train --models "Random Forest" "XGBoost" "LightGBM"
```

### Step 3: Make Predictions
```powershell
# Using test data
python predictor.py

# Or batch predictions
python main.py predict --input data/synthetic_farmers.csv --output results/predictions.csv
```

## Option 3: Python Scripts

### Generate data and train in one go:
```powershell
# Generate data
python data_generator.py

# Train model
python model_trainer.py

# Make predictions
python predictor.py
```

## Verify Installation

Test if everything works:
```powershell
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, streamlit; print('All packages installed successfully!')"
```

## Common Commands

### Launch Web Interface
```powershell
streamlit run app.py
```

### Generate 1000 farmer records
```powershell
python main.py generate --sample-size 1000
```

### Train specific models
```powershell
python main.py train --models "XGBoost" "LightGBM"
```

### Calculate credit scores only
```powershell
python main.py score --input data/farmers.csv --output data/scored.csv
```

## Project Files

- `app.py` - Streamlit web interface (recommended)
- `main.py` - Command-line interface
- `data_generator.py` - Generate synthetic farmer data
- `model_trainer.py` - Train ML models
- `predictor.py` - Make predictions
- `scoring_engine.py` - Calculate credit scores
- `feature_engineering.py` - Create features
- `config.py` - Configuration settings

## Getting Help

```powershell
# Main CLI help
python main.py --help

# Command-specific help
python main.py generate --help
python main.py train --help
python main.py predict --help
```

## Troubleshooting

**Error: "streamlit not found"**
```powershell
pip install streamlit
```

**Error: "No module named X"**
```powershell
pip install -r requirements.txt
```

**Error: Virtual environment not activating**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

## Next Steps

1. Read `README.md` for detailed documentation
2. Explore the web interface at `http://localhost:8501`
3. Review the generated data in `data/synthetic_farmers.csv`
4. Check model performance in the Analytics Dashboard
5. Try evaluating single applications

---

**Questions?** Check the comprehensive `README.md` file for detailed information.
