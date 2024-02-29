import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer

# Page layout
st.set_page_config(layout="wide")

# Title
st.title("Advanced Data Analysis and Visualization Tool")

# Sidebar
st.sidebar.title("Settings")
data_option = st.sidebar.radio("Select data option:", ("Upload CSV file", "Generate random data"))
plot_option = st.sidebar.multiselect("Select plot types:", ["Histogram", "Box plot", "Scatter plot", "Pair plot", "Heatmap", "Countplot", "3D plot"])
enable_ml = st.sidebar.checkbox("Enable Machine Learning")

# Main content
if data_option == "Upload CSV file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load data
        @st.cache
        def load_data(file):
            return pd.read_csv(file)

        df = load_data(uploaded_file)

else:
    st.sidebar.subheader("Generate Random Data")
    data_type = st.sidebar.radio("Select data type:", ("Classification", "Regression"))
    if data_type == "Classification":
        n_samples = st.sidebar.slider("Number of samples:", min_value=100, max_value=1000, step=100)
        n_features = st.sidebar.slider("Number of features:", min_value=1, max_value=10, step=1)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        df = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(1, n_features+1)])
        df['Target'] = y
    else:
        n_samples = st.sidebar.slider("Number of samples:", min_value=100, max_value=1000, step=100)
        n_features = st.sidebar.slider("Number of features:", min_value=1, max_value=10, step=1)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        df = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(1, n_features+1)])
        df['Target'] = y

# Data exploration
if "Data Exploration" in st.session_state:
    st.subheader("Data Exploration")
    st.write(df.head())
    st.write("Shape of the data:", df.shape)
    st.write("Summary statistics:")
    st.write(df.describe())
else:
    st.session_state["Data Exploration"] = True

# Data cleaning
if "Data Cleaning" in st.session_state:
    st.subheader("Data Cleaning")
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.write("Missing values:")
        st.write(missing_values)
        fill_method = st.selectbox("Select fill method:", ["Drop missing values", "Fill with mean", "Fill with median", "KNN"])
        if fill_method == "Drop missing values":
            df.dropna(inplace=True)
        elif fill_method == "Fill with mean":
            df.fillna(df.mean(), inplace=True)
        elif fill_method == "Fill with median":
            df.fillna(df.median(), inplace=True)
        elif fill_method == "KNN":
            imputer = KNNImputer()
            df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            st.write(df_filled)
else:
    st.session_state["Data Cleaning"] = True

# Data visualization
if "Data Visualization" in st.session_state:
    st.subheader("Data Visualization")
    if plot_option:
        for plot_type in plot_option:
            st.write(plot_type)
            if plot_type == "Histogram":
                column = st.selectbox("Select column for histogram:", df.columns)
                plt.figure(figsize=(8, 6))
                sns.histplot(df[column], kde=True)
                st.pyplot()
            elif plot_type == "Box plot":
                column = st.selectbox("Select column for box plot:", df.columns)
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df[column])
                st.pyplot()
            elif plot_type == "Scatter plot":
                x_column = st.selectbox("Select X-axis column:", df.columns)
                y_column = st.selectbox("Select Y-axis column:", df.columns)
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=x_column, y=y_column)
                st.pyplot()
            elif plot_type == "Pair plot":
                plt.figure(figsize=(10, 8))
                sns.pairplot(df)
                st.pyplot()
            elif plot_type == "Heatmap":
                plt.figure(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
                st.pyplot()
            elif plot_type == "Countplot":
                column = st.selectbox("Select column for countplot:", df.columns)
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df, x=column)
                st.pyplot()
            elif plot_type == "3D plot":
                x_column = st.selectbox("Select X-axis column:", df.columns)
                y_column = st.selectbox("Select Y-axis column:", df.columns)
                z_column = st.selectbox("Select Z-axis column:", df.columns)
                fig = go.Figure(data=[go.Scatter3d(x=df[x_column], y=df[y_column], z=df[z_column], mode='markers')])
                st.plotly_chart(fig)
else:
    st.session_state["Data Visualization"] = True

# Machine Learning
if enable_ml:
    st.subheader("Machine Learning")
    if st.checkbox("Train a basic machine learning model"):
        X = df.drop(columns=["Target"])
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("R-squared:", r2_score(y_test, y_pred))

# Advanced Plot Customization
st.subheader("Advanced Plot Customization")
if st.checkbox("Customize plot aesthetics"):
    # Example: Customize scatter plot with selected options
    marker_style = st.selectbox("Marker style:", ["o", "s", "D"])
    line_style = st.selectbox("Line style:", ["-", "--", "-.", ":"])
    color = st.color_picker("Marker color:", "#ff5733")

    plt.figure(figsize=(8, 6))
    plt.plot(df[x_column], df[y_column], marker=marker_style, linestyle=line_style, color=color)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("Customized Scatter Plot")
    st.pyplot()

# Data Filtering
st.subheader("Data Filtering")
if st.checkbox("Filter data"):
    # Example: Filter data based on user input for a specific column
    filter_column = st.selectbox("Select column to filter:", df.columns)
    filter_value = st.text_input("Enter filter value:")
    filtered_df = df[df[filter_column] == filter_value]
    st.write(filtered_df)

# Missing Value Imputation
st.subheader("Missing Value Imputation")
if st.checkbox("Impute missing values"):
    # Example: Impute missing values using mean
    missing_cols = df.columns[df.isnull().any()]
    if missing_cols.any():
        impute_method = st.selectbox("Select imputation method:", ["Mean", "Median", "KNN"])
        if impute_method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif impute_method == "Median":
            df.fillna(df.median(), inplace=True)
        elif impute_method == "KNN":
            imputer = KNNImputer()
            df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            st.write(df_filled)
