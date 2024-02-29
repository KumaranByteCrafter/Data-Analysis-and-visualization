import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from scipy import stats

def preprocess_data(df):
    # Fill missing values and remove duplicates
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def create_plot(df, plot_type, x_axis, y_axis=None):
    try:
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
        elif plot_type == 'Violin Plot':
            fig = px.violin(df, x=x_axis, y=y_axis)
        elif plot_type == 'Boxen Plot':
            fig = px.box(df, x=x_axis, y=y_axis)
        elif plot_type == 'Scatter Matrix':
            fig = px.scatter_matrix(df)
        elif plot_type == 'Pair Density Plot':
            fig = px.density_contour(df, x=x_axis, y=y_axis)
        return fig
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
highlighted_title = "<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;'>" \
                    "Comprehensive Data Analysis and Visualization Tool" \
                    "</div>"
st.markdown(highlighted_title, unsafe_allow_html=True)

# Adding developer credit
st.markdown("<div style='text-align: center;'>Developed by Kumaran R</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)

    # Data Exploration
    st.sidebar.subheader("Data Exploration")
    if st.sidebar.checkbox("View Data Head"):
        st.subheader("Data Head (First 5 Rows)")
        st.dataframe(df.head())
    if st.sidebar.checkbox("View Data Description"):
        st.subheader("Data Description (Statistical Summary)")
        st.dataframe(df.describe())
    if st.sidebar.checkbox("View Data Types"):
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))
    if st.sidebar.checkbox("View Missing Data"):
        st.subheader("Missing Data Report")
        missing_data = df.isnull().sum()
        st.dataframe(missing_data.to_frame('Missing Values'))
    if st.sidebar.checkbox("View Duplicate Data"):
        st.subheader("Duplicate Data Report")
        duplicate_data = df.duplicated().sum()
        st.write(f"Duplicate Rows: {duplicate_data}")
    if st.sidebar.checkbox("View Unique Values"):
        pd.set_option('display.max_colwidth', None)
        unique_column = st.sidebar.selectbox("Select Column for Unique Values", df.columns)
        st.subheader(f"Unique Values in '{unique_column}' Column")
        unique_values_df = pd.DataFrame(df[unique_column].unique(), columns=[unique_column])
        max_table_height = 400
        st.dataframe(unique_values_df, width=800, height=max_table_height)
    if st.sidebar.checkbox("View Correlation Matrix"):
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        st.write(numeric_df.corr())
    if st.sidebar.checkbox("View Value Counts"):
        st.subheader("Value Counts")
        value_count_column = st.sidebar.selectbox("Select Column for Value Counts", df.columns)
        st.write(df[value_count_column].value_counts())
    if st.sidebar.checkbox("Detect Outliers"):
        st.subheader("Outlier Detection")
        outlier_column = st.sidebar.selectbox("Select Column for Outlier Detection", df.select_dtypes(include=np.number).columns)
        z_scores = np.abs(stats.zscore(df[outlier_column]))
        threshold = 3
        outliers = np.where(z_scores > threshold)[0]
        if len(outliers) > 0:
            st.write("Outliers detected in the selected column:")
            st.write(df.iloc[outliers])
        else:
            st.write("No outliers detected in the selected column.")

    # Visualization
    st.sidebar.subheader("Data Visualization")
    plot_types = ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Pie Chart', 'Area Chart', 'Heatmap', 'Violin Plot', 'Boxen Plot', 'Scatter Matrix', 'Pair Density Plot']
    plot_choice = st.sidebar.selectbox("Choose plot type", plot_types)
    x_axis = st.selectbox('Select X-axis', df.columns)
    y_axis = None
    if plot_choice not in ['Histogram', 'Pie Chart', 'Heatmap']:
        y_axis = st.selectbox('Select Y-axis', df.columns)
    if st.button('Generate Plot'):
        st.subheader(f"{plot_choice}")
        fig = create_plot(df, plot_choice, x_axis, y_axis)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
