# Demand Forecasting & Business Planning

A production-ready demand forecasting solution with Prophet and XGBoost models, Flask REST API, and Power BI integration.

## Features

- **Historical Dataset**: Uses 5 years of retail sales data (900K+ records)
- **Automated Setup**: One-command data preparation
- **Weekly Aggregation**: Reduces noise for more stable forecasts
- **Dual Models**: Prophet for seasonality + XGBoost for feature-rich predictions
- **Flask API**: RESTful endpoints for real-time predictions
- **Power BI Ready**: CSV exports for easy visualization
- **Docker Support**: Containerized deployment

## Dataset

The project uses a comprehensive retail sales dataset with:

- **Records**: 900,000+ daily sales transactions
- **Period**: 5 years of historical data (2013-2018)
- **Stores**: Multiple store locations
- **Items**: 50+ different products
- **Features**: Date, store, item, sales volume

## Project Structure

```
demand-forecast/
├─ data/
│  ├─ train.csv               # Raw dataset
│  └─ sample_data.csv         # Processed for modeling
├─ notebooks/
│  └─ quick_explore.ipynb     # Exploratory analysis
├─ src/
│  ├─ download_data.py        # Dataset downloader
│  ├─ data_prep.py            # Load & resample to weekly
│  ├─ features.py             # Lag features for XGBoost
│  ├─ train_prophet.py        # Train & save Prophet model
│  ├─ train_xgboost.py        # Train & save XGBoost model
│  ├─ evaluate.py             # Export forecasts to CSV
│  └─ app.py                  # Flask prediction API
├─ outputs/                   # CSV forecasts for Power BI
├─ models/                    # Saved .joblib models
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

**Option A: Use included sample data** (Quick start)
```bash
# Sample data already included in data/sample_data.csv
# Skip to Step 3
```

**Option B: Download full dataset** (Recommended for production)
```bash
python src/download_data.py
```

This will download and process the complete dataset (~28 MB), aggregating sales across all stores and items to create `data/sample_data.csv` with columns: `ds`, `y`, `promo`.

### Step 3: Train Models

```bash
# Train Prophet model
python src/train_prophet.py

# Train XGBoost model
python src/train_xgboost.py
```

Models are saved to `models/` as `.joblib` files.

### Step 4: Evaluate & Export for Power BI

```bash
python src/evaluate.py
```

Outputs CSV files to `outputs/`:
- `prophet_forecast.csv` - Prophet predictions
- `xgb_forecast.csv` - XGBoost predictions

**Import these CSVs into Power BI** to build dashboards with actual vs predicted visualizations.

### Step 5: Run Flask API

```bash
# Development mode
python src/app.py

# Production mode (recommended)
gunicorn -b 0.0.0.0:5000 src.app:app --workers=2 --timeout=120
```

API available at `http://localhost:5000`

## API Endpoints

### `GET /`
Returns API info

### `GET /predict?model=prophet&horizon_weeks=12`
Returns forecast predictions

**Query Parameters:**
- `model` - `prophet` or `xgb` (default: `prophet`)
- `horizon_weeks` - Number of weeks to forecast (default: `12`)

**Example:**
```bash
curl "http://localhost:5000/predict?model=xgb&horizon_weeks=8"
```

**Response:**
```json
{
  "model": "xgb",
  "predictions": [
    {"ds": "2018-01-01", "y_pred": 2847.52},
    {"ds": "2018-01-08", "y_pred": 2895.18},
    ...
  ]
}
```

## Docker Deployment

```bash
# Build image
docker build -t demand-forecast .

# Run container
docker run -p 5000:5000 demand-forecast
```

Access API at `http://localhost:5000`

## Power BI Integration

### 1. Generate Forecasts

```bash
python src/evaluate.py
```

### 2. Import to Power BI Desktop

1. **Get Data → Text/CSV**
2. Load `outputs/prophet_forecast.csv` and/or `outputs/xgb_forecast.csv`
3. **Create visualizations:**
   - Line chart: `ds` (x-axis) vs `y` and `yhat` (y-axis)
   - KPI cards: Display MAE, RMSE metrics
   - Slicers: Date range filters

### 3. Refresh Data

- Re-run `python src/evaluate.py` to update CSVs
- Click **"Refresh"** in Power BI to reload

## Model Details

### Prophet
- **Input:** Weekly aggregated sales with `promo` regressor
- **Captures:** Weekly/yearly seasonality, promotional spikes, trend changes
- **Use case:** Long-term forecasts (3-12 months)
- **Output:** Predictions with confidence intervals
- **Advantage:** Learns from 5 years of historical patterns

### XGBoost
- **Features:** Lags (1,2,3,4,8,12 weeks), rolling mean, day-of-week, month, promo flag
- **Training:** Early stopping on validation set
- **Use case:** Short-term forecasts (1-3 months)
- **Output:** Point predictions with feature importance
- **Advantage:** Captures complex item/store interactions

## Data Options

### Option 1: Aggregate All Stores & Items (Default)

```python
# In src/download_data.py (default)
df = prepare_data(train_path)
```

Aggregates sales across all stores and items for total demand.

### Option 2: Single Store & Item

```python
# In src/download_data.py (uncomment in __main__)
df = prepare_single_store_item(store_id=1, item_id=1)
```

Forecasts demand for a specific store-item combination.

## Customization

### Use Your Own Data

Replace `data/sample_data.csv` with your CSV containing:
- `ds` - Date column (YYYY-MM-DD format)
- `y` - Target variable (sales, demand, etc.)
- `promo` - Binary promotional indicator (0/1)

Then re-run training scripts.

### Tune Hyperparameters

**Prophet** (`src/train_prophet.py`):
```python
m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05  # Adjust trend flexibility (0.001-0.5)
)
```

**XGBoost** (`src/train_xgboost.py`):
```python
params = {
    'eta': 0.1,           # Learning rate (0.01-0.3)
    'max_depth': 6,       # Tree depth (3-10)
    'num_boost_round': 200  # Number of trees (50-500)
}
```

## Requirements

```
pandas==2.2.2
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.1.6
prophet==1.2.1
flask==2.3.2
gunicorn==20.1.0
joblib==1.3.2
matplotlib==3.8.1
kaggle==1.6.6
opendatasets==0.1.22
```

Install: `pip install -r requirements.txt`

## Jupyter Notebook

Explore the data interactively:

```bash
jupyter notebook notebooks/quick_explore.ipynb
```

## Complete Workflow

```
1. Install dependencies  →  pip install -r requirements.txt
2. Prepare data          →  python src/download_data.py (optional)
3. Train models          →  python src/train_prophet.py + train_xgboost.py
4. Evaluate & export     →  python src/evaluate.py (outputs CSVs)
5. Visualize in Power BI →  Import CSVs to Power BI
6. Deploy API            →  Docker or Gunicorn for production
```

## Troubleshooting

### "Model not found" error
- Run training scripts first: `python src/train_prophet.py`

### Import errors when running scripts
- Ensure you're running from repo root (not inside `src/`)
- Check virtual environment is activated

### Power BI won't load CSV
- Verify file exists in `outputs/` folder
- Try using absolute path in Power BI import

### Dataset download issues
- Check internet connection
- Verify API credentials if using external data sources
- Use included sample data as fallback

## Dataset Statistics

After downloading and processing:
- **Training period**: 5 years of historical data
- **Total records**: 900,000+ daily transactions
- **Aggregated weekly**: ~260 weeks
- **Features**: Date, sales volume, promotional indicators
- **Data quality**: Clean, no missing values

## Why This Dataset?

✅ **Historical data** - Multi-year retail sales patterns
✅ **Large scale** - 900K+ records for robust learning
✅ **Captures trends** - Seasonality and promotional effects
✅ **Clean** - Pre-processed and validated
✅ **Business relevant** - Typical retail forecasting scenario

## License

MIT License - free to use and modify.

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- Models train without errors
- CSVs export correctly
- Document any dataset changes

## Support

Open an issue on GitHub for questions or bugs.

## Acknowledgments

- Models: Facebook Prophet, XGBoost
- Data processing: Pandas, NumPy
- API: Flask, Gunicorn
