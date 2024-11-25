import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(page_title='California Housing Crisis App')

# Load Dataset
df = pd.read_csv("housing.csv")

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Introduction", "Exploration", "Visualization", "Prediction", "Conclusion"],
    icons=["house", "search", "bar-chart-line", "lightbulb", "check-circle"],
    default_index=0,
    orientation="horizontal",
)

if selected == 'Introduction':
    st.title("California Housing Crisis 🏠")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_path = Image.open("housing_image.jpg")
        st.image(image_path, width=400)

    st.write("""
    ## Introduction
    Housing affordability and availability are pressing issues in California, impacting millions of residents and the state's economy. This app explores California housing price data to uncover trends, correlations, and potential solutions for combating the housing crisis.

    ## Objective
    This app aims to:
    - Explore factors influencing housing prices.
    - Analyze trends in affordability and availability.
    - Provide actionable insights and potential solutions to address the housing crisis.

    ## Key Features
    - Visualization of housing price trends and influential factors.
    - Analysis of correlations between demographics, geography, and housing costs.
    - Predictive modeling for housing prices.
    """)

elif selected == 'Exploration':
    st.title("Data Exploration 🔍")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset Head", "Dataset Tail", "Description", "Missing Values", "Generate Report"])

    with tab1:
        st.subheader("Head of the Dataset")
        st.dataframe(df.head())

    with tab2:
        st.subheader("Tail of the Dataset")
        st.dataframe(df.tail())

    with tab3:
        st.subheader("Description of the Dataset")
        st.dataframe(df.describe())

    with tab4:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum() / len(df) * 100
        total_missing = missing_data.sum().round(2)
        st.write(missing_data)
        if total_missing == 0.0:
            st.success("There are no missing values!")
        else:
            st.warning("There are missing values.")

elif selected == 'Visualization':
    st.title("Data Visualization 📊")
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Geographic Heatmap", "Correlation Heatmap", "Feature Relationships"])

    with tab1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Housing Prices")
        st.pyplot(fig)

    with tab2:
        st.subheader("Geographic Heatmap of Median House Value")
        st.map(df[['latitude', 'longitude']])

    with tab3:
        st.subheader("Correlation Heatmap")
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    with tab4:
        st.subheader("Relationships Between Features")
        x_feature = st.selectbox("Select X-axis Feature:", df.columns)
        y_feature = st.selectbox("Select Y-axis Feature:", df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
        ax.set_title(f"Relationship Between {x_feature} and {y_feature}")
        st.pyplot(fig)

elif selected == "Prediction":
    st.title("Predicting Housing Prices 💡")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = st.multiselect("Select Features for Prediction", numeric_columns)
    target = st.selectbox("Select Target Variable", ["median_house_value"])

    if features:
        X = df[features]
        y = df[target]
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)
        
        st.write("### Prediction Results")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")

elif selected == 'Conclusion':
    st.title("Conclusion 🏁")
    st.write("""
    ### Key Insights:
    1. **Housing Affordability**: Rising housing costs in California are closely linked to population density and proximity to urban centers.
    2. **Influential Factors**: Features like household income, location, and proximity to amenities significantly impact housing prices.

    ### Proposed Solutions:
    1. **Affordable Housing Initiatives**: Increase funding for affordable housing projects and incentivize developers.
    2. **Zoning Reforms**: Encourage high-density housing developments through zoning changes.
    3. **Public Transportation Investments**: Improve transportation infrastructure to connect remote areas with urban job markets.
    """)
