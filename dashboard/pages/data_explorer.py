"""
Data Explorer Page

Visualize and analyze the demand forecasting data.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def show():
    st.title("📈 Data Explorer")
    st.markdown("Explore and visualize the demand forecasting dataset.")

    # Load data
    try:
        from src.features.store import load_features

        df = load_features()
        st.success(f"Loaded {len(df):,} records with {len(df.columns)} features")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("Run the feature engineering pipeline first.")
        return

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Time Series", "Distributions", "Correlations"]
    )

    with tab1:
        show_overview(df)

    with tab2:
        show_time_series(df)

    with tab3:
        show_distributions(df)

    with tab4:
        show_correlations(df)


def show_overview(df: pd.DataFrame):
    """Show data overview."""
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
        st.metric("Total Columns", len(df.columns))

    with col2:
        st.metric(
            "Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}"
        )
        st.metric("Unique Products", df["product_id"].nunique())

    with col3:
        st.metric("Unique Stores", df["store_id"].nunique())
        st.metric("Unique Categories", df["category_id"].nunique())

    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Column info
    st.subheader("Column Statistics")
    st.dataframe(df.describe(), use_container_width=True)


def show_time_series(df: pd.DataFrame):
    """Show time series visualizations."""
    st.subheader("Time Series Analysis")

    # Aggregate by date
    daily_demand = df.groupby("date")["target_demand"].mean().reset_index()

    # Line chart
    fig = px.line(
        daily_demand,
        x="date",
        y="target_demand",
        title="Average Daily Demand Over Time",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Average Demand")
    st.plotly_chart(fig, use_container_width=True)

    # By category
    col1, col2 = st.columns(2)

    with col1:
        category_demand = (
            df.groupby("category_id")["target_demand"].mean().reset_index()
        )
        fig = px.bar(
            category_demand,
            x="category_id",
            y="target_demand",
            title="Average Demand by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        store_demand = df.groupby("store_id")["target_demand"].mean().reset_index()
        fig = px.bar(
            store_demand,
            x="store_id",
            y="target_demand",
            title="Average Demand by Store",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weekly pattern
    if "day_of_week" in df.columns:
        dow_demand = df.groupby("day_of_week")["target_demand"].mean().reset_index()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_demand["day_name"] = dow_demand["day_of_week"].map(lambda x: dow_names[x])

        fig = px.bar(
            dow_demand,
            x="day_name",
            y="target_demand",
            title="Average Demand by Day of Week",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_distributions(df: pd.DataFrame):
    """Show feature distributions."""
    st.subheader("Feature Distributions")

    # Select feature
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    selected_feature = st.selectbox(
        "Select Feature",
        numeric_cols,
        index=numeric_cols.index("target_demand")
        if "target_demand" in numeric_cols
        else 0,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x=selected_feature,
            nbins=50,
            title=f"Distribution of {selected_feature}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    # Demand by promotion/holiday
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df,
            x="promotion_flag",
            y="target_demand",
            title="Demand: Promotion vs No Promotion",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="holiday_flag",
            y="target_demand",
            title="Demand: Holiday vs Regular Day",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_correlations(df: pd.DataFrame):
    """Show feature correlations."""
    st.subheader("Feature Correlations")

    # Select features for correlation
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Limit to important features for readability
    important_cols = [
        "target_demand",
        "historical_sales",
        "price",
        "promotion_flag",
        "holiday_flag",
        "economic_index",
        "day_of_week",
        "month",
    ]
    default_cols = [c for c in important_cols if c in numeric_cols]

    selected_cols = st.multiselect(
        "Select Features for Correlation", numeric_cols, default=default_cols[:8]
    )

    if len(selected_cols) >= 2:
        corr_matrix = df[selected_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with target
        if "target_demand" in selected_cols:
            st.subheader("Top Correlations with Target Demand")
            target_corr = (
                corr_matrix["target_demand"]
                .drop("target_demand")
                .sort_values(key=abs, ascending=False)
            )

            fig = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation="h",
                title="Feature Correlations with Target Demand",
            )
            fig.update_layout(xaxis_title="Correlation", yaxis_title="Feature")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least 2 features to show correlations.")
