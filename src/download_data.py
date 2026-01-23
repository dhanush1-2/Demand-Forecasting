# src/download_data.py
"""
Download and prepare dataset for demand forecasting
Dataset: https://www.kaggle.com/datasets/programmer3/demand-forecasting-dataset
"""
import os
import pandas as pd
from pathlib import Path
import zipfile
import subprocess


def download_kaggle_dataset():
    """
    Download demand forecasting dataset from Kaggle

    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token"
       - Save kaggle.json to ~/.kaggle/kaggle.json
       - chmod 600 ~/.kaggle/kaggle.json
    """
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    data_dir.mkdir(exist_ok=True)

    # Check if data already exists
    if (data_dir / 'train.csv').exists():
        print("Dataset already exists. Skipping download.")
        return data_dir / 'train.csv'

    print("Downloading dataset from Kaggle...")

    try:
        # Download using Kaggle API
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', 'programmer3/demand-forecasting-dataset',
            '-p', str(data_dir),
            '--unzip'
        ]
        subprocess.run(cmd, check=True)

        print(f"Dataset downloaded to {data_dir}")

        # Look for the train file
        possible_paths = [
            data_dir / 'train.csv',
            data_dir / 'Train.csv',
            data_dir / 'historical_demand.csv'
        ]

        for path in possible_paths:
            if path.exists():
                # Rename to standard train.csv if needed
                if path.name != 'train.csv':
                    path.rename(data_dir / 'train.csv')
                return data_dir / 'train.csv'

        # If no standard file found, look for any CSV
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files and csv_files[0].name != 'train.csv':
            csv_files[0].rename(data_dir / 'train.csv')
            return data_dir / 'train.csv'
        elif csv_files:
            return csv_files[0]

        return data_dir / 'train.csv'

    except subprocess.CalledProcessError:
        print("\nError: Kaggle API not configured properly.")
        print("Please follow these steps:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get your API token from https://www.kaggle.com/account")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return None


def prepare_data(train_path=None):
    """
    Prepare and aggregate the dataset

    Original columns: date, store, item, sales
    Prepared columns: ds, y, promo
    """
    data_dir = Path(__file__).resolve().parents[1] / 'data'

    if train_path is None:
        train_path = data_dir / 'train.csv'

    if not train_path.exists():
        print(f"Error: {train_path} not found")
        print("Run download_kaggle_dataset() first")
        return None

    print(f"Loading data from {train_path}...")
    df = pd.read_csv(train_path)

    # Show dataset info
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Check for date column variations
    date_col = None
    for col in ['date', 'Date', 'DATE', 'ds']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print("Error: No date column found")
        return None

    df['date'] = pd.to_datetime(df[date_col])
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Check for store and item columns
    if 'store' in df.columns:
        print(f"Number of stores: {df['store'].nunique()}")
    if 'item' in df.columns:
        print(f"Number of items: {df['item'].nunique()}")

    # Aggregate all stores and items (you can filter specific store/item if needed)
    print("\nAggregating data across all stores and items...")

    # Determine sales column
    sales_col = None
    for col in ['sales', 'Sales', 'SALES', 'y', 'demand', 'Demand']:
        if col in df.columns:
            sales_col = col
            break

    if sales_col is None:
        print("Error: No sales/demand column found")
        return None

    # Group by date and sum sales
    df_agg = df.groupby('date')[sales_col].sum().reset_index()
    df_agg.columns = ['ds', 'y']

    # Add a promo flag (simulated based on sales spikes)
    # You can replace this with actual promo data if available
    df_agg['promo'] = 0
    rolling_mean = df_agg['y'].rolling(window=7, center=True).mean()
    df_agg.loc[df_agg['y'] > rolling_mean * 1.3, 'promo'] = 1
    df_agg['promo'] = df_agg['promo'].fillna(0).astype(int)

    # Save processed data
    output_path = data_dir / 'sample_data.csv'
    df_agg.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    print(f"Final data shape: {df_agg.shape}")
    print(f"Columns: {df_agg.columns.tolist()}")

    return df_agg


def prepare_single_store_item(store_id=1, item_id=1):
    """
    Prepare data for a specific store and item (alternative to aggregation)

    Args:
        store_id: Store number
        item_id: Item number
    """
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    train_path = data_dir / 'train.csv'

    if not train_path.exists():
        print("Error: train.csv not found")
        return None

    print(f"Loading data for store {store_id}, item {item_id}...")
    df = pd.read_csv(train_path)

    # Find date column
    date_col = next((col for col in ['date', 'Date', 'DATE', 'ds'] if col in df.columns), None)
    if date_col:
        df['date'] = pd.to_datetime(df[date_col])

    # Find sales column
    sales_col = next((col for col in ['sales', 'Sales', 'SALES', 'y', 'demand'] if col in df.columns), None)

    # Filter for specific store and item
    if 'store' in df.columns and 'item' in df.columns:
        df_filtered = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
        df_filtered = df_filtered[['date', sales_col]].rename(columns={'date': 'ds', sales_col: 'y'})
    else:
        print("Warning: Store/item columns not found. Using all data.")
        df_filtered = df[['date', sales_col]].rename(columns={'date': 'ds', sales_col: 'y'})

    # Add promo flag
    df_filtered['promo'] = 0
    rolling_mean = df_filtered['y'].rolling(window=7, center=True).mean()
    df_filtered.loc[df_filtered['y'] > rolling_mean * 1.3, 'promo'] = 1
    df_filtered['promo'] = df_filtered['promo'].fillna(0).astype(int)

    # Save
    output_path = data_dir / 'sample_data.csv'
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved store {store_id}, item {item_id} data to {output_path}")
    print(f"Shape: {df_filtered.shape}")

    return df_filtered


if __name__ == "__main__":
    print("=" * 60)
    print("Demand Forecasting - Dataset Setup")
    print("=" * 60)

    # Download dataset
    train_path = download_kaggle_dataset()

    if train_path and train_path.exists():
        print("\n" + "=" * 60)
        print("Preparing data...")
        print("=" * 60)

        # Option 1: Aggregate all stores and items
        df = prepare_data(train_path)

        # Option 2: Use specific store and item (uncomment to use)
        # df = prepare_single_store_item(store_id=1, item_id=1)

        if df is not None:
            print("\n" + "=" * 60)
            print("Setup complete! You can now train models.")
            print("=" * 60)
            print("\nNext steps:")
            print("1. python src/train_prophet.py")
            print("2. python src/train_xgboost.py")
            print("3. python src/evaluate.py")
    else:
        print("\nSetup incomplete. Please configure Kaggle API and try again.")
