
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Food Delivery Time Prediction", layout="centered", page_icon="🍔")

# Sidebar navigation
st.sidebar.title("🍔 Food Delivery Dashboard")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction 🔮"])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Food_Delivery_Times.csv")
    return df

df = load_data()

# Page 1: Introduction
if page == "Introduction 📘":
    st.title("🚴 Food Delivery Time Prediction")
    st.markdown("## 🎯 Problem Statement")
    st.markdown("""
        Food delivery companies struggle with accurately estimating delivery times.
        Inaccurate estimates reduce customer satisfaction and can hurt business.
        This app aims to **predict delivery time** based on factors like distance, traffic, weather, and driver experience
        using a **linear regression model**.
    """)
    st.image("food.jpg")

    st.markdown("## 📁 Dataset Overview")
    rows = st.slider("Preview rows", 5, 30, 10)
    st.dataframe(df.head(rows))

    st.markdown("### 🔎 Missing Values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values")
    else:
        st.warning("⚠️ Some columns have missing values and will be dropped for modeling.")

    st.markdown("### 📊 Summary Statistics")
    if st.button("Show Summary"):
        st.dataframe(df.describe())

# Page 2: Visualization
elif page == "Visualization 📊":
    st.title("📊 Data Insights")
    df_viz = df.dropna()

    st.markdown("### 🚗 Delivery Vehicle Type Distribution")
    vehicle_counts = df_viz["Vehicle_Type"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(vehicle_counts, labels=vehicle_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Distribution of Delivery Vehicle Types")
    st.pyplot(fig1)

    st.markdown("### 📏 Avg Delivery Time by Distance Segment")
    bins = [0, 5, 10, 15, 20, 25]
    labels = ["0-5km", "5-10km", "10-15km", "15-20km", "20-25km"]
    df_viz["Distance_Segment"] = pd.cut(df_viz["Distance_km"], bins=bins, labels=labels)
    avg_by_segment = df_viz.groupby("Distance_Segment")["Delivery_Time_min"].mean().reset_index()

    fig2, ax2 = plt.subplots()
    sns.barplot(x="Distance_Segment", y="Delivery_Time_min", data=avg_by_segment, ax=ax2)
    ax2.set_xlabel("Distance Segment")
    ax2.set_ylabel("Average Delivery Time (min)")
    ax2.set_title("Avg Delivery Time by Distance Segment")
    st.pyplot(fig2)

    st.markdown("### 📌 How does distance relate to delivery time?")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_viz, x="Distance_km", y="Delivery_Time_min", hue="Traffic_Level", ax=ax)
    ax.set_title("Delivery Time vs. Distance colored by Traffic Level")
    st.pyplot(fig)

    st.markdown("### 📉 Correlation Heatmap")
    df_numeric = df_viz.select_dtypes(include=np.number)
    fig3, ax3 = plt.subplots()
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# Page 3: Prediction
elif page == "Prediction 🔮":
    st.title("🔮 Predicting Delivery Time")
    st.markdown("""
        Using the cleaned dataset, we build a **linear regression model** to predict delivery time.
        The goal is to help food delivery businesses estimate customer wait time more accurately.
    """)

    df_model = df.dropna().copy()
    le_weather = LabelEncoder()
    le_traffic = LabelEncoder()
    le_time = LabelEncoder()
    le_vehicle = LabelEncoder()

    df_model["Weather"] = le_weather.fit_transform(df_model["Weather"])
    df_model["Traffic_Level"] = le_traffic.fit_transform(df_model["Traffic_Level"])
    df_model["Time_of_Day"] = le_time.fit_transform(df_model["Time_of_Day"])
    df_model["Vehicle_Type"] = le_vehicle.fit_transform(df_model["Vehicle_Type"])

    features = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", 
                "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
    target = "Delivery_Time_min"

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write(f"**MAE**: {mean_absolute_error(y_test, predictions):.2f}")
    st.write(f"**MSE**: {mean_squared_error(y_test, predictions):.2f}")
    st.write(f"**R² Score**: {r2_score(y_test, predictions):.3f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Delivery Time")
    ax.set_ylabel("Predicted Delivery Time")
    ax.set_title("Actual vs Predicted Delivery Time")
    st.pyplot(fig)

    st.subheader("📌 Key Insights")
    st.markdown("""
    - **Feature Impact:** Distance, Traffic Level, and Preparation Time were the most influential features in predicting delivery time.
    - **Model Fit:** The model achieves an R² score of ~0.77, indicating decent predictive power, but improvements are possible.
    - **Real-World Use:** Businesses can use this model to estimate delivery ETAs and improve customer satisfaction. More complex models or live traffic inputs could enhance future predictions.
    """)
