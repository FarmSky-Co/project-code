"""
Main entry point for command-line usage
Provides a simple CLI for common tasks
"""

import argparse
import os
from data_generator import generate_and_save_data
from model_trainer import CreditModelTrainer
from feature_engineering import FeatureEngineer
from predictor import CreditPredictor
from scoring_engine import CreditScoringEngine
from config import *
import pandas as pd


def generate_data(args):
    """Generate synthetic data"""
    print(f"Generating {args.sample_size} farmer records...")
    df = generate_and_save_data()
    print(f"✓ Data saved to {DATA_DIR}/{SYNTHETIC_DATA_FILE}")
    

def train_model(args):
    """Train ML models"""
    # Check if data exists
    data_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
    if not os.path.exists(data_path):
        print("Error: No training data found. Run 'generate' first.")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.prepare_for_modeling(df)
    
    feature_cols = engineer.get_all_model_features()
    
    print("Training models...")
    trainer = CreditModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, feature_cols, use_smote=not args.no_smote
    )
    
    models_to_train = args.models if args.models else None
    results = trainer.train_all_models(X_train, X_test, y_train, y_test, models_to_train)
    
    print("\nModel Comparison:")
    print(results[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])
    
    print("\nSaving model...")
    trainer.save_model()
    print("✓ Training complete!")


def predict(args):
    """Make predictions"""
    if not os.path.exists(f"{MODELS_DIR}/{TRAINED_MODEL_FILE}"):
        print("Error: No trained model found. Run 'train' first.")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    print(f"Loading data from {args.input}...")
    predictor = CreditPredictor()
    
    output = args.output if args.output else f"{RESULTS_DIR}/predictions.csv"
    predictions = predictor.batch_predict(args.input, output)
    
    print(f"\n✓ Predictions saved to {output}")


def score(args):
    """Calculate credit scores"""
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Calculating credit scores...")
    scorer = CreditScoringEngine()
    df_scored = scorer.score_applications(df)
    
    summary = scorer.get_score_summary(df_scored)
    
    print("\nScore Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if args.output:
        df_scored.to_csv(args.output, index=False)
        print(f"\n✓ Scored data saved to {args.output}")


def web_interface(args):
    """Launch web interface"""
    import subprocess
    
    print("Launching Streamlit web interface...")
    print("Press Ctrl+C to stop the server.")
    
    subprocess.run(['streamlit', 'run', 'app.py'])


def main():
    parser = argparse.ArgumentParser(
        description='Farmer Credit Scoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data
  python main.py generate --sample-size 1000
  
  # Train models
  python main.py train --models "Random Forest" "XGBoost"
  
  # Make predictions
  python main.py predict --input data/new_applications.csv --output results/predictions.csv
  
  # Calculate credit scores
  python main.py score --input data/farmers.csv --output data/scored.csv
  
  # Launch web interface
  python main.py web
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate command
    parser_gen = subparsers.add_parser('generate', help='Generate synthetic data')
    parser_gen.add_argument('--sample-size', type=int, default=SYNTHETIC_SAMPLE_SIZE,
                           help='Number of farmer records to generate')
    parser_gen.set_defaults(func=generate_data)
    
    # Train command
    parser_train = subparsers.add_parser('train', help='Train ML models')
    parser_train.add_argument('--models', nargs='+', 
                             choices=['Logistic Regression', 'Random Forest', 
                                     'Gradient Boosting', 'XGBoost', 'LightGBM'],
                             help='Specific models to train')
    parser_train.add_argument('--no-smote', action='store_true',
                             help='Disable SMOTE for class balancing')
    parser_train.set_defaults(func=train_model)
    
    # Predict command
    parser_pred = subparsers.add_parser('predict', help='Make predictions')
    parser_pred.add_argument('--input', required=True, help='Input CSV file')
    parser_pred.add_argument('--output', help='Output CSV file')
    parser_pred.set_defaults(func=predict)
    
    # Score command
    parser_score = subparsers.add_parser('score', help='Calculate credit scores')
    parser_score.add_argument('--input', required=True, help='Input CSV file')
    parser_score.add_argument('--output', help='Output CSV file')
    parser_score.set_defaults(func=score)
    
    # Web command
    parser_web = subparsers.add_parser('web', help='Launch web interface')
    parser_web.set_defaults(func=web_interface)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
