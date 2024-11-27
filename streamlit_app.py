import streamlit as st
import pandas as pd
import numpy as np
import codecs
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

# MLFlow/DagHub Integration
import dagshub
dagshub.init(repo_owner='sufyanw', repo_name='blank-app-1', mlflow=True)

# Fill in missing values in dataset with the median value
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Introduction", "Visualization", "Prediction", "MLFlow", "Explainable AI", "Conclusion"],
    icons=["house", "bar-chart-line", "lightbulb", "cloud", "robot", "check-circle"],
    default_index=0,
    orientation="horizontal",
)

# Pages
if selected == 'Introduction':
    st.title("Housing Crisis 🏠")
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

elif selected == 'Visualization':
    st.title("Data Visualization 📊")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Distribution", "Geographic Heatmap", "Correlation Heatmap", "Feature Relationships", "Generate Report"])

    with tab1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Housing Prices")
        st.pyplot(fig)

    with tab2:
        st.subheader("Geographic Heatmap of Median House Value")

        # Define color based on price range
        def price_to_color(value):
            if value < 150000:
                return [255, 0, 0]  # Red
            elif value < 300000:
                return [0, 0, 255]  # Blue
            else:
                return [0, 255, 0]  # Green

        # Add color and size columns to the dataframe
        df['color'] = df['median_house_value'].apply(price_to_color)
        df['size'] = (df['median_house_value'] - df['median_house_value'].min()) / (
            df['median_house_value'].max() - df['median_house_value'].min()
        ) * 100  # Normalize size to scale between 0 and 100

        # Create a Pydeck map
        import pydeck as pdk

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="size",
            radius_scale=10,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=df['latitude'].mean(),
            longitude=df['longitude'].mean(),
            zoom=6,
            pitch=0,
        )

        map = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Price: {median_house_value}"},
        )

        st.pydeck_chart(map)

    with tab3:
        st.subheader("Correlation Heatmap")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_columns].corr()
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

    with tab5:
        if st.button("Generate Report"):
            def read_html_report(file_path):
                with codecs.open(file_path, 'r', encoding="utf-8") as f:
                    return f.read()
            
            html_report = read_html_report('housing_report.html')
            
            st.title("Streamlit Quality Report")
            st.components.v1.html(html_report, height=1000, scrolling=True)


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

elif selected == "MLFlow":
    st.title("MLFlow Integration 🌩️")
    st.write("""
    ## Model Tracking with MLFlow
    This app integrates MLFlow through DagHub to track the following:
    - Experiment runs and parameters.
    - Performance metrics (MAE, R² Score).
    - Model artifacts for reproducibility.
    
    ### How to Access
    Visit the [MLFlow Dashboard](https://dagshub.com/sufyanw/blank-app-1) for detailed experiment tracking.
    """)

elif selected == "Explainable AI":
    st.title("Explainable AI 🔎🤖")
    st.write("""
    ## Explainable AI for Model Insights
    To make the predictions transparent and interpretable, we use **SHAP (SHapley Additive exPlanations)**.

    ### Key Features:
    - Understand feature contributions to each prediction.
    - Visualize the global importance of features.

    ### Example Insights:
    Select an instance to analyze its SHAP explanation below.
    """)
    # Placeholder for SHAP visualizations (requires more setup in actual implementation)
    st.info("SHAP visualization coming soon!")

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