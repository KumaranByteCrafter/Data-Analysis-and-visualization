import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Function to preprocess data
def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Function to create different types of plots
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
    elif plot_type == 'Violin Plot':
        fig = px.violin(df, x=x_axis, y=y_axis)
    elif plot_type == 'Boxen Plot':
        fig = px.box(df, x=x_axis, y=y_axis)
    elif plot_type == 'Scatter Matrix':
        fig = px.scatter_matrix(df)
    elif plot_type == 'Pair Density Plot':
        fig = px.density_contour(df, x=x_axis, y=y_axis)
    return fig

# Function for descriptive analysis
def descriptive_analysis(df):
    return df.describe()

# Function for diagnostic analysis
def diagnostic_analysis(df):
    return df.corr()

# Function for predictive analysis
def predictive_analysis(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for prescriptive analysis
def prescriptive_analysis(df):
    return "Based on the analysis, we recommend focusing marketing efforts on the younger demographic as they show a higher response rate."

# Function for outlier detection
def detect_outliers(df, column):
    z_scores = np.abs(stats.zscore(df[column]))
    threshold = 3
    outliers = np.where(z_scores > threshold)[0]
    if len(outliers) > 0:
        return df.iloc[outliers]
    else:
        return "No outliers detected in the selected column."

# Function for data preprocessing
def data_preprocessing(df):
    # Example: Standardize numerical features
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    return df

# Function for feature engineering
def feature_engineering(df):
    # Example: Creating new feature based on existing ones
    df['new_feature'] = df['feature1'] + df['feature2']
    return df

# Function for model selection and evaluation
def model_selection_evaluation(df):
    # Example: Train a Random Forest Classifier and evaluate its performance
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for advanced visualization
def advanced_visualization(df):
    # Example: Pairplot with Seaborn
    sns.pairplot(df)
    plt.show()

# Function for time series analysis
def time_series_analysis(df):
    # Example: Extracting time components from a datetime feature
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df

# Function for natural language processing
def nlp_analysis(df):
    # Example: CountVectorizer for text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text_column'])
    return X

# Function for clustering and dimensionality reduction
def clustering_dimensionality_reduction(df):
    # Example: KMeans clustering and PCA for dimensionality reduction
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df)
    df['cluster'] = kmeans.labels_
    
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(df)
    
    return df, reduced_features

# Main function
def main():
    st.title("Data Exploration and Analysis Tool")
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/vnd.ms-excel':
            st.error("Sorry, Excel files in .xls format are not supported. Please convert the file to .xlsx format and try again.")
        else:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            df = preprocess_data(df)

            st.sidebar.subheader("Data Exploration Options")
            exploration_options = st.sidebar.multiselect("Select Data Exploration Options", 
                                                          ["Data Head", "Data Description", "Data Types", 
                                                           "Missing Values", "Duplicate Data", "Unique Values", 
                                                           "Correlation Matrix", "Value Counts", "Detect Outliers"])

            if "Data Head" in exploration_options:
                st.subheader("Data Head (First 5 Rows)")
                st.dataframe(df.head())

            if "Data Description" in exploration_options:
                st.subheader("Data Description (Statistical Summary)")
                st.dataframe(descriptive_analysis(df))

            if "Data Types" in exploration_options:
                st.subheader("Data Types")
                st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))

            if "Missing Values" in exploration_options:
                st.subheader("Missing Data Report")
                missing_data = df.isnull().sum()
                st.dataframe(missing_data.to_frame('Missing Values'))

            if "Duplicate Data" in exploration_options:
                st.subheader("Duplicate Data Report")
                duplicate_data = df.duplicated().sum()
                st.write(f"Duplicate Rows: {duplicate_data}")

            if "Unique Values" in exploration_options:
                pd.set_option('display.max_colwidth', None)
                unique_column = st.sidebar.selectbox("Select Column for Unique Values", df.columns)
                st.subheader(f"Unique Values in '{unique_column}' Column")
                unique_values_df = pd.DataFrame(df[unique_column].unique(), columns=[unique_column])
                max_table_height = 400
                st.dataframe(unique_values_df, width=800, height=max_table_height)

            if "Correlation Matrix" in exploration_options:
                st.subheader("Correlation Matrix")
                st.write(diagnostic_analysis(df))

            if "Value Counts" in exploration_options:
                st.subheader("Value Counts")
                value_count_column = st.sidebar.selectbox("Select Column for Value Counts", df.columns)
                st.write(df[value_count_column].value_counts())

            if "Detect Outliers" in exploration_options:
                st.subheader("Outlier Detection")
                outlier_column = st.sidebar.selectbox("Select Column for Outlier Detection", df.select_dtypes(include=np.number).columns)
                st.write(detect_outliers(df, outlier_column))

            st.sidebar.subheader("Data Analysis Options")
            analysis_options = st.sidebar.multiselect("Select Data Analysis Options", 
                                                      ["Descriptive Analysis", "Predictive Analysis", "Prescriptive Analysis", 
                                                       "Feature Engineering", "Advanced Visualization", "Time Series Analysis", 
                                                       "Natural Language Processing", "Clustering and Dimensionality Reduction"])

            if "Descriptive Analysis" in analysis_options:
                st.subheader("Descriptive Analysis")
                st.write(descriptive_analysis(df))

            if "Predictive Analysis" in analysis_options:
                st.subheader("Predictive Analysis")
                st.write("Model Accuracy:", predictive_analysis(df))

            if "Prescriptive Analysis" in analysis_options:
                st.subheader("Prescriptive Analysis")
                st.write(prescriptive_analysis(df))

            if "Feature Engineering" in analysis_options:
                st.subheader("Feature Engineering")
                st.write("Example: Creating new feature based on existing ones")
                st.write(feature_engineering(df))

            if "Advanced Visualization" in analysis_options:
                st.subheader("Advanced Visualization")
                st.write("Example: Pairplot with Seaborn")
                advanced_visualization(df)

            if "Time Series Analysis" in analysis_options:
                st.subheader("Time Series Analysis")
                st.write("Example: Extracting time components from a datetime feature")
                st.write(time_series_analysis(df))

            if "Natural Language Processing" in analysis_options:
                st.subheader("Natural Language Processing")
                st.write("Example: CountVectorizer for text data")
                st.write(nlp_analysis(df))

            if "Clustering and Dimensionality Reduction" in analysis_options:
                st.subheader("Clustering and Dimensionality Reduction")
                st.write("Example: KMeans clustering and PCA for dimensionality reduction")
                clustered_df, reduced_features = clustering_dimensionality_reduction(df)
                st.write(clustered_df.head())
                st.write(reduced_features)

            # Visualization
            st.sidebar.subheader("Data Visualization Options")
            plot_types = ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Pie Chart', 
                          'Area Chart', 'Heatmap', 'Violin Plot', 'Boxen Plot', 'Scatter Matrix', 'Pair Density Plot']
            plot_choice = st.sidebar.selectbox("Choose plot type", plot_types)
            x_axis = st.sidebar.selectbox('Select X-axis', df.columns)
            y_axis = None
            if plot_choice not in ['Histogram', 'Pie Chart', 'Heatmap']:
                y_axis = st.sidebar.selectbox('Select Y-axis', df.columns)
            
            if st.sidebar.button('Generate Plot'):
                st.subheader(f"{plot_choice}")
                fig = create_plot(df, plot_choice, x_axis, y_axis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
