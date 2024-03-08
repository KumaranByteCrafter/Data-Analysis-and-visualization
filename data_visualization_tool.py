import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats

def preprocess_data(df):
    # Fill missing values with forward fill method for simplicity
    df.fillna(method='ffill', inplace=True)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df

def remove_outliers(df, z_thresh=3):
    # Remove outliers based on Z-score
    z = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z < z_thresh).all(axis=1)]
    return df

def create_plot(df, plot_type, x_axis, y_axis=None):
    if plot_type == 'Bar Chart':
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif plot_type == 'Line Chart':
        fig = px.line(df, x=x_axis, y=y_axis)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=x_axis, y=y_axis)
    elif plot_type == 'Histogram':
        fig = px.histogram(df, x=x_axis)
    elif plot_type == 'Box Plot':
        fig = px.box(df, x=x_axis, y=y_axis)
    elif plot_type == 'Pie Chart':
        fig = px.pie(df, names=x_axis, values=y_axis)
    elif plot_type == 'Heatmap':
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
    else:
        fig = px.area(df, x=x_axis, y=y_axis)
    return fig

# Streamlit App Configuration
st.set_page_config(page_title="Advanced Data Analysis Tool")
st.title("Advanced Data Analysis Tool")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)

    if st.sidebar.checkbox("Treat Missing Values"):
        df.fillna(df.mean(), inplace=True)  # Simple imputation with mean
        st.write("Missing values treated with mean for numerical columns.")

    if st.sidebar.checkbox("Remove Outliers"):
        df = remove_outliers(df)
        st.write("Outliers removed based on Z-score method.")

    st.sidebar.subheader("Data Exploration")
    if st.sidebar.checkbox("Show Dataframe"):
        st.write(df)

    if st.sidebar.checkbox("Variable Identification"):
        st.write(df.dtypes)

    if st.sidebar.checkbox("Univariate Analysis"):
        st.write(df.describe())

    if st.sidebar.checkbox("Bivariate Analysis"):
        st.write("Select specific plots from the 'Generate Plot' section.")

    # Visualization
    plot_types = ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Pie Chart', 'Heatmap', 'Area Chart']
    plot_choice = st.sidebar.selectbox("Choose Plot Type", plot_types)
    x_axis = st.selectbox('Select X-axis', options=df.columns)
    y_axis = None
    if plot_choice not in ['Histogram', 'Pie Chart', 'Heatmap']:
        y_axis = st.selectbox('Select Y-axis', options=df.columns)
    if st.button('Generate Plot'):
        fig = create_plot(df, plot_choice, x_axis, y_axis)
        st.plotly_chart(fig, use_container_width=True)
