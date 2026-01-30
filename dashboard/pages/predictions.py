"""
Predictions Page

Make real-time demand predictions using the API.
"""

import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import plotly.express as px
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")





def show():
    st.title("🔮 Demand Predictions")
    st.markdown("Make real-time demand predictions using the trained model.")
    
    # Check API status
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code != 200:
            st.error("API is not responding correctly. Start the API first.")
            return
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Start it with: `uvicorn src.api.main:app --reload`")
        return
    
    st.success("✅ Connected to API")
    
    # Tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        show_single_prediction()
    
    with tab2:
        show_batch_prediction()


def show_single_prediction():
    """Single prediction form."""
    st.subheader("Make a Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_date = st.date_input(
            "Prediction Date",
            value=date.today() + timedelta(days=1)
        )
        product_id = st.number_input("Product ID", min_value=1, value=1)
        category_id = st.number_input("Category ID", min_value=1, value=1)
        store_id = st.number_input("Store ID", min_value=1, value=1)
        historical_sales = st.number_input("Historical Sales", min_value=0.0, value=100.0)
    
    with col2:
        price = st.number_input("Price", min_value=0.01, value=25.0)
        promotion_flag = st.selectbox("Promotion Active?", [0, 1], index=0)
        holiday_flag = st.selectbox("Is Holiday?", [0, 1], index=0)
        economic_index = st.number_input("Economic Index", value=100.0)
    
    if st.button("🔮 Predict Demand", type="primary"):
        # Prepare request
        payload = {
            "prediction_date": str(prediction_date),
            "product_id": product_id,
            "category_id": category_id,
            "store_id": store_id,
            "historical_sales": historical_sales,
            "price": price,
            "promotion_flag": promotion_flag,
            "holiday_flag": holiday_flag,
            "economic_index": economic_index
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("### Prediction Result")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Demand", f"{result['predicted_demand']:.2f}")
                with col2:
                    st.metric("Model Used", result["model_used"].upper())
                with col3:
                    st.metric("Confidence", result["confidence"].capitalize())
                
            else:
                st.error(f"Prediction failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error: {e}")


def show_batch_prediction():
    """Batch prediction interface."""
    st.subheader("Batch Predictions")
    st.markdown("Generate predictions for multiple scenarios.")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Start Date", value=date.today())
        n_days = st.slider("Number of Days", 1, 30, 7)
    
    with col2:
        product_id = st.number_input("Product ID", min_value=1, value=1, key="batch_product")
        store_id = st.number_input("Store ID", min_value=1, value=1, key="batch_store")
    
    with col3:
        category_id = st.number_input("Category ID", min_value=1, value=1, key="batch_category")
        base_price = st.number_input("Base Price", min_value=0.01, value=25.0)
    
    if st.button("📊 Generate Forecast", type="primary"):
        predictions = []
        
        progress = st.progress(0)
        
        for i in range(n_days):
            pred_date = start_date + timedelta(days=i)
            
            payload = {
                "prediction_date": str(pred_date),
                "product_id": product_id,
                "category_id": category_id,
                "store_id": store_id,
                "historical_sales": 100.0,
                "price": base_price,
                "promotion_flag": 0,
                "holiday_flag": 0,
                "economic_index": 100.0
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    predictions.append({
                        "date": pred_date,
                        "predicted_demand": result["predicted_demand"]
                    })
            except:
                pass
            
            progress.progress((i + 1) / n_days)
        
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Chart
            fig = px.line(
                df,
                x="date",
                y="predicted_demand",
                title=f"Demand Forecast for Product {product_id}, Store {store_id}",
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Demand")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Demand", f"{df['predicted_demand'].mean():.2f}")
            with col2:
                st.metric("Min Demand", f"{df['predicted_demand'].min():.2f}")
            with col3:
                st.metric("Max Demand", f"{df['predicted_demand'].max():.2f}")
            
            # Data table
            st.dataframe(df, use_container_width=True)
        else:
            st.error("No predictions generated. Check API connection.")
