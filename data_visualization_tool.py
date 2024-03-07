import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Setup
st.title("Data Analysis and Modeling Tool")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        st.sidebar.warning("Converted Excel file to CSV")

    # Side bar options
    sidebar_options = st.sidebar.selectbox("Select an option", ("Overview", "Data Cleaning", "Visualization", "EDA", "Business Analytics", "Model Building", "Dashboard"))

    # Variables to store analysis results
    overview_results = None
    missing_values_results = None
    duplicate_values_results = None
    visualization_results = None
    correlation_matrix_results = None
    pairplot_results = None
    business_analytics_results = None
    model_results = None

    # Overview
    if sidebar_options == "Overview":
        st.header("Dataset Overview")
        overview_results = {
            "Number of rows": df.shape[0],
            "Number of columns": df.shape[1],
            "Column data types": df.dtypes,
            "Preview of dataset": df.head()
        }
        st.write("Number of rows:", overview_results["Number of rows"])
        st.write("Number of columns:", overview_results["Number of columns"])
        st.write("Column data types:", overview_results["Column data types"])
        st.write("Preview of dataset:", overview_results["Preview of dataset"])

    # Data Cleaning
    elif sidebar_options == "Data Cleaning":
        st.header("Data Cleaning")
        # Handling missing values
        if st.checkbox("Handle Missing Values"):
            method = st.selectbox("Select method:", ("Drop missing values", "Impute missing values"))
            if method == "Drop missing values":
                df.dropna(inplace=True)
                missing_values_results = "Missing values dropped successfully!"
                st.success(missing_values_results)
            else:
                st.warning("Imputation method not implemented yet!")

        # Handling duplicate values
        if st.checkbox("Handle Duplicate Values"):
            df.drop_duplicates(inplace=True)
            duplicate_values_results = "Duplicate values removed successfully!"
            st.success(duplicate_values_results)

    # Visualization
    elif sidebar_options == "Visualization":
        st.header("Data Visualization")
        # Automatic plot selection based on data type and unique values
        st.write("Automatic plot selection based on data type and unique values")
        for column in df.columns:
            if df[column].dtype == 'object':
                st.write("Count plot for:", column)
                st.write(sns.countplot(x=column, data=df))
                st.pyplot()
            elif len(df[column].unique()) < 10:
                st.write("Histogram for:", column)
                st.write(sns.histplot(x=column, data=df))
                st.pyplot()
            else:
                st.write("KDE plot for:", column)
                st.write(sns.kdeplot(x=column, data=df))
                st.pyplot()

    # EDA
    elif sidebar_options == "EDA":
        st.header("Advanced EDA and Data Analytics")
        # Correlation matrix
        if st.checkbox("Correlation Matrix"):
            st.write("Correlation Matrix:")
            correlation_matrix_results = df.corr()
            st.write(correlation_matrix_results)

        # Pairplot
        if st.checkbox("Pairplot"):
            st.write("Pairplot:")
            pairplot_results = sns.pairplot(df)
            st.pyplot()

    # Business Analytics
    elif sidebar_options == "Business Analytics":
        st.header("Business Analytics Support")
        # Sales by category
        if st.checkbox("Sales by Category"):
            sales_by_category = df.groupby('Category')['Sales'].sum()
            st.write("Sales by Category:")
            st.write(sales_by_category)

            # Plot interactive bar chart
            fig = px.bar(sales_by_category, x=sales_by_category.index, y='Sales', title='Sales by Category')
            st.plotly_chart(fig)

        # Profit by sub-category
        if st.checkbox("Profit by Sub-Category"):
            profit_by_subcategory = df.groupby('Sub-Category')['Profit'].sum()
            st.write("Profit by Sub-Category:")
            st.write(profit_by_subcategory)

            # Plot interactive bar chart
            fig = px.bar(profit_by_subcategory, x=profit_by_subcategory.index, y='Profit', title='Profit by Sub-Category')
            st.plotly_chart(fig)

    # Model Building
    elif sidebar_options == "Model Building":
        st.header("Model Building")
        task = st.radio("Select task:", ("Regression", "Classification"))

        X = df.dropna().drop('target_column', axis=1)  # Adjust 'target_column' according to your dataset
        y = df.dropna()['target_column']  # Adjust 'target_column' according to your dataset

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task == "Classification":
            model_name = st.selectbox("Select model:", ("Random Forest", "Naive Bayes", "Decision Tree", "KNN", "SVM"))
            if model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "KNN":
                model = KNeighborsClassifier()
            else:
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            model_results = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Classification Report": classification_report(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred)
            }
            st.write("Accuracy:", model_results["Accuracy"])
            st.write("Classification Report:", model_results["Classification Report"])
            st.write("Confusion Matrix:", model_results["Confusion Matrix"])

        else:
            model_name = st.selectbox("Select model:", ("Linear Regression", "Random Forest", "Decision Tree", "KNN", "SVR"))
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_name == "KNN":
                model = KNeighborsRegressor()
            else:
                model = SVR()

            model.fit(X_train, y_train)
            y_pred = model
