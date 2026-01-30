"""
Demand Forecasting Dashboard

Main Streamlit application entry point.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📈 Data Explorer", "🔮 Predictions", "📊 Model Performance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Demand Forecasting MLOps**\n\n"
        "An end-to-end ML pipeline for retail demand prediction."
    )
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📈 Data Explorer":
        from dashboard.pages import data_explorer
        data_explorer.show()
    elif page == "🔮 Predictions":
        from dashboard.pages import predictions
        predictions.show()
    elif page == "📊 Model Performance":
        from dashboard.pages import model_performance
        model_performance.show()


def show_home():
    """Home page content."""
    import os
    import requests
    
    # Get API URL from environment (for Docker) or default to localhost
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    st.markdown('<p class="main-header">Demand Forecasting Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Demand Forecasting MLOps Dashboard. This application provides:
    
    - **Data Explorer**: Visualize and analyze historical demand data
    - **Predictions**: Make real-time demand predictions
    - **Model Performance**: View model metrics and comparisons
    """)
    
    # Quick stats
    st.markdown("### Quick Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        from src.features.store import load_features
        df = load_features()
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Features", f"{len(df.columns)}")
        
        with col3:
            n_products = df["product_id"].nunique()
            st.metric("Products", f"{n_products}")
        
        with col4:
            n_stores = df["store_id"].nunique()
            st.metric("Stores", f"{n_stores}")
            
    except Exception as e:
        st.warning(f"Could not load data: {e}")
    
    # System status
    st.markdown("### System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**API Status**")
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API is running")
            else:
                st.error(f"❌ API returned error: {response.status_code}")
        except Exception as e:
            st.warning(f"⚠️ API not reachable: {e}")
    
    with col2:
        st.markdown("**Model Status**")
        try:
            response = requests.get(f"{api_url}/health/model", timeout=2)
            if response.status_code == 200:
                model_info = response.json()
                st.success(f"✅ Model loaded: {model_info['model_name']}")
            else:
                st.error("❌ Model not loaded")
        except Exception as e:
            st.info(f"ℹ️ Start API to see model status")



if __name__ == "__main__":
    main()
