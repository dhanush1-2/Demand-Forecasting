"""
Model Performance Page

Display model metrics and comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import pickle

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def show():
    st.title("📊 Model Performance")
    st.markdown("View model metrics, feature importance, and comparisons.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Model Metrics", "Feature Importance", "Predictions vs Actual"])
    
    with tab1:
        show_model_metrics()
    
    with tab2:
        show_feature_importance()
    
    with tab3:
        show_predictions_vs_actual()


def show_model_metrics():
    """Display model metrics."""
    st.subheader("Model Evaluation Metrics")
    
    # Try to load saved metrics or compute them
    try:
        from src.features.store import load_features, create_train_test_split
        from src.models.evaluation import calculate_metrics
        from src.utils.config import get_paths
        
        paths = get_paths()
        
        # Load features
        df = load_features()
        train_df, test_df = create_train_test_split(df)
        
        target_col = "target_demand"
        exclude_cols = ["date", target_col]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Load models and evaluate
        metrics_data = []
        
        # LightGBM
        lgb_path = paths["models"] / "lightgbm_model.pkl"
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            metrics["model"] = "LightGBM"
            metrics_data.append(metrics)
        
        # XGBoost
        xgb_path = paths["models"] / "xgboost_model.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            metrics["model"] = "XGBoost"
            metrics_data.append(metrics)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df = metrics_df.set_index("model")
            
            # Display metrics
            st.dataframe(
                metrics_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r"),
                use_container_width=True
            )
            
            # Bar chart comparison
            fig = go.Figure()
            for metric in ["mae", "rmse", "mape"]:
                if metric in metrics_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric.upper(),
                        x=metrics_df.index,
                        y=metrics_df[metric]
                    ))
            
            fig.update_layout(
                title="Model Comparison",
                barmode="group",
                xaxis_title="Model",
                yaxis_title="Error Value"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trained models found.")
            
    except Exception as e:
        st.error(f"Error loading metrics: {e}")


def show_feature_importance():
    """Display feature importance."""
    st.subheader("Feature Importance")
    
    try:
        from src.utils.config import get_paths
        
        paths = get_paths()
        
        # Try LightGBM first
        lgb_path = paths["models"] / "lightgbm_model.pkl"
        xgb_path = paths["models"] / "xgboost_model.pkl"
        
        model = None
        model_name = None
        feature_cols = None
        
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            feature_cols = model_data.get("feature_columns", [])
            model_name = "LightGBM"
        elif xgb_path.exists():
            with open(xgb_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            feature_cols = model_data.get("feature_columns", [])
            model_name = "XGBoost"
        
        if model and feature_cols:
            # Get feature importance
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importance
            }).sort_values("importance", ascending=True)
            
            # Top N features
            n_features = st.slider("Number of Features to Show", 10, len(feature_cols), 20)
            top_features = importance_df.tail(n_features)
            
            fig = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top {n_features} Features - {model_name}"
            )
            fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
            st.plotly_chart(fig, use_container_width=True)
            
            # Full table
            with st.expander("View All Features"):
                st.dataframe(
                    importance_df.sort_values("importance", ascending=False),
                    use_container_width=True
                )
        else:
            st.warning("No model found or no feature columns saved.")
            
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")


def show_predictions_vs_actual():
    """Show predictions vs actual values."""
    st.subheader("Predictions vs Actual")
    
    try:
        from src.features.store import load_features, create_train_test_split
        from src.utils.config import get_paths
        
        paths = get_paths()
        
        # Load data
        df = load_features()
        train_df, test_df = create_train_test_split(df)
        
        target_col = "target_demand"
        exclude_cols = ["date", target_col]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        dates = test_df["date"]
        
        # Load model
        lgb_path = paths["models"] / "lightgbm_model.pkl"
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                model_data = pickle.load(f)
            model = model_data["model"]
            y_pred = model.predict(X_test)
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                "date": dates.values,
                "actual": y_test.values,
                "predicted": y_pred
            })
            
            # Line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison_df["date"],
                y=comparison_df["actual"],
                name="Actual",
                line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                x=comparison_df["date"],
                y=comparison_df["predicted"],
                name="Predicted",
                line=dict(color="red", dash="dash")
            ))
            fig.update_layout(
                title="Actual vs Predicted Demand",
                xaxis_title="Date",
                yaxis_title="Demand"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                comparison_df,
                x="actual",
                y="predicted",
                title="Actual vs Predicted (Scatter)"
            )
            # Add diagonal line
            max_val = max(comparison_df["actual"].max(), comparison_df["predicted"].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash", color="gray")
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals
            comparison_df["residual"] = comparison_df["actual"] - comparison_df["predicted"]
            
            fig = px.histogram(
                comparison_df,
                x="residual",
                nbins=50,
                title="Residual Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No trained model found.")
            
    except Exception as e:
        st.error(f"Error: {e}")
