import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Data Analysis Tool", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={'Get Help': None, 'Report a bug': None, 'About': None})

# Function to preprocess data
def preprocess_data(df):
    # Fill missing values and remove duplicates
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Function to create plots
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
        numeric_df = df.select_dtypes(include=[np.number])
        fig = px.imshow(numeric_df.corr(), text_auto=True)
    elif plot_type == 'Area Chart':
        fig = px.area(df, x=x_axis, y=y_axis)
    return fig

# Function for EDA
def perform_eda(df):
    st.sidebar.subheader("Exploratory Data Analysis (EDA)")

    # Load and read dataset
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)

    # Data Exploration
    st.sidebar.subheader("Data Exploration")

    options = {
        "View Data Head": df.head(),
        "View Data Description": df.describe(),
        "View Data Types": df.dtypes.astype(str).to_frame('Data Type'),
        "View Missing Data": df.isnull().sum().to_frame('Missing Values'),
        "View Duplicate Data": df.duplicated().sum(),
        "Show Data Information": df.info(),
        "Show Column Names": df.columns.tolist(),
        "Check for Missing Values": df.isnull().sum(),
        "Check for Duplicate Values": df.nunique()
    }

    for option in options:
        if st.sidebar.checkbox(option):
            st.subheader(option)
            st.write(options[option])

# Inject custom Bootstrap-like CSS
custom_css = """
<style>
    /* Custom CSS for Streamlit app */
    .reportview-container .main .block-container{
        padding: 2rem;
    }
    .stButton>button {
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        cursor: pointer;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

highlighted_title = "<div style='background-color: #007bff; color: white; padding: 10px; border-radius: 5px; text-align: center;'>" \
                    "Comprehensive Data Analysis and Visualization Tool" \
                    "</div>"
st.markdown(highlighted_title, unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-bottom: 20px;'>Developed by Kumaran R</div>", unsafe_allow_html=True)

# Main function
def main():
    # Sidebar
    perform_eda(df)

    # Data Visualization
    st.sidebar.subheader("Data Visualization")
    plot_types = ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Pie Chart', 'Area Chart', 'Heatmap']
    plot_choice = st.sidebar.selectbox("Choose plot type", plot_types)
    x_axis = st.selectbox('Select X-axis', df.columns)
    y_axis = None
    if plot_choice not in ['Histogram', 'Pie Chart', 'Heatmap']:
        y_axis = st.selectbox('Select Y-axis', df.columns)
    if st.button('Generate Plot'):
        fig = create_plot(df, plot_choice, x_axis, y_axis)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
