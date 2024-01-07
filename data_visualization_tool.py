import streamlit as st
import pandas as pd
import plotly.express as px
def preprocess_data(df):
    # Fill missing values and remove duplicates
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
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
    elif plot_type == 'Area Chart':
        fig = px.area(df, x=x_axis, y=y_axis)
    elif plot_type == 'Heatmap':
        fig = px.imshow(df.corr(), text_auto=True)
    return fig
# Streamlit app layout
st.title('Comprehensive Data Analysis and Visualization Tool')
highlighted_text = "<div style='background-color:yellow; padding: 10px; border-radius: 5px;'>" \
                   "I'm Kumaran, developed this project" \
                   "</div>"
st.markdown(highlighted_text, unsafe_allow_html=True)
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
        unique_column = st.sidebar.selectbox("Select Column for Unique Values", df.columns)
        st.subheader(f"Unique Values in '{unique_column}' Column")
        st.write(df[unique_column].unique())
    # Visualization
    st.sidebar.subheader("Data Visualization")
    plot_types = ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Pie Chart', 'Area Chart', 'Heatmap']
    plot_choice = st.sidebar.selectbox("Choose plot type", plot_types)
    x_axis = st.selectbox('Select X-axis', df.columns)
    y_axis = None
    if plot_choice not in ['Histogram', 'Pie Chart', 'Heatmap']:
        y_axis = st.selectbox('Select Y-axis', df.columns)
    if st.button('Generate Plot'):
        st.subheader(f"{plot_choice}")
        fig = create_plot(df, plot_choice, x_axis, y_axis)
        st.plotly_chart(fig, use_container_width=True)
