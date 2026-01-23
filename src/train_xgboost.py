# src/train_xgboost.py
import xgboost as xgb
import joblib
from pathlib import Path
from src.data_prep import load_data, resample_weekly
from src.features import create_lag_features, train_test_split_time
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models'
MODEL_PATH.mkdir(exist_ok=True)

def train_and_save(df=None):
    if df is None:
        df = load_data()
    df = resample_weekly(df)
    df = create_lag_features(df)
    train, test = train_test_split_time(df, test_weeks=12)
    features = [c for c in train.columns if c not in ('ds','y')]
    dtrain = xgb.DMatrix(train[features], train['y'])
    dtest = xgb.DMatrix(test[features], test['y'])
    params = {
        'objective':'reg:squarederror',
        'eval_metric':'rmse',
        'eta':0.1,
        'max_depth':6
    }
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'eval')], early_stopping_rounds=20, verbose_eval=False)
    joblib.dump({'model':bst, 'features':features}, MODEL_PATH / 'xgb_model.joblib')
    # metrics
    preds = bst.predict(dtest)
    mae = mean_absolute_error(test['y'], preds)
    rmse = mean_squared_error(test['y'], preds, squared=False)
    print(f"Saved XGBoost model to {MODEL_PATH/'xgb_model.joblib'} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return bst, features

if __name__ == "__main__":
    train_and_save()
