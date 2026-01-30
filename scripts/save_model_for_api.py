"""Save a trained model for API use."""

import sys
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.ingestion import load_raw_data
from src.data.transformation import transform_data
from src.features.engineering import create_features
from src.features.store import save_features, load_features, create_train_test_split
from src.models.lightgbm_model import LightGBMModel
from src.utils.config import get_paths


def main():
    paths = get_paths()
    feature_file = paths["features"] / "features_latest.parquet"
    
    # Check if features exist, if not create them
    if not feature_file.exists():
        print("Features not found. Creating features...")
        
        # Load and transform data
        print("  Loading raw data...")
        df = load_raw_data()
        
        print("  Transforming data...")
        df = transform_data(df)
        
        print("  Engineering features...")
        df = create_features(df)
        
        print("  Saving features...")
        save_features(df)
    
    print("Loading features...")
    df = load_features()
    
    print("Creating train/test split...")
    train_df, test_df = create_train_test_split(df)
    
    # Separate features and target
    target_col = "target_demand"
    exclude_cols = ["date", target_col]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    print(f"Training with {len(feature_cols)} features...")
    
    print("Training LightGBM model...")
    model = LightGBMModel()
    model.fit(X_train, y_train)
    
    print("Saving model for API...")
    model_path = paths["models"] / "lightgbm_model.pkl"
    
    model_data = {
        "model": model.model,
        "feature_columns": list(X_train.columns),
        "model_name": "lightgbm",
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Features: {len(X_train.columns)}")
    print("Done! You can now start the API.")


if __name__ == "__main__":
    main()
