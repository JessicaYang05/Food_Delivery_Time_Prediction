import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
import pickle

# Page config
st.set_page_config(page_title="Food Delivery Time Prediction", layout="centered", page_icon="🍔")

# Sidebar navigation
st.sidebar.title("🍔 Food Delivery Dashboard")
page = st.sidebar.selectbox("Select Page", [
    "Introduction 📘", 
    "Visualization 📊", 
    "Prediction 🔮",
    "Explainability 🤔",
    "Model Tracker 📊",
    "Conclusion 📌",
    "What-If Simulator 🔁"
])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Food_Delivery_Times.csv")
    return df

df = load_data()

# Page 1: Introduction
if page == "Introduction 📘":
    st.title("🚴 Food Delivery Time Prediction")
    st.markdown("## 🌟 Problem Statement")
    st.markdown("""
        Food delivery companies struggle with accurately estimating delivery times.
        Inaccurate estimates reduce customer satisfaction and can hurt business.
        This app aims to **predict delivery time** based on factors like distance, traffic, weather, and driver experience
        using different machine learning models.
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

    st.markdown("### 🛏️ Avg Delivery Time by Distance Segment")
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
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    st.title("🔮 Predicting Delivery Time")
    st.markdown("""
        Use different models to predict delivery time and compare their performance.
    """)

    # Handle missing values
    df_model = df.dropna().copy()

    # Encode categoricals
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

    model_choice = st.selectbox("Choose your model", ["Linear Regression", "Decision Tree", "K-Nearest Neighbors"])

    with mlflow.start_run():
        if model_choice == "Linear Regression":
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
        elif model_choice == "Decision Tree":
            # Classification setup
            df_model["FastDelivery"] = (df_model["Delivery_Time_min"] <= 30).astype(int)
            target = "FastDelivery"

            X = df_model[features]
            y = df_model[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # UI for depth
            max_depth = st.number_input("Enter the maximum depth of the decision tree", 1, 20, value=5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Metrics
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)

            # Show metrics
            st.subheader("🧮 Decision Tree Prediction Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Decision Tree' f1-Score", f"{f1*100:.1f}%", "vs last run")
            col2.metric("Accuracy", f"{acc*100:.1f}%", "vs last run")
            col3.metric("Precision", f"{precision*100:.1f}%", "vs last run")

            # Visualization
            st.subheader("🌳 Decision Tree Visualization")
            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
            plot_tree(model, feature_names=features, class_names=["Slow", "Fast"], filled=True, rounded=True, fontsize=10)
            st.pyplot(fig_tree)

        elif model_choice == "K-Nearest Neighbors":
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            import seaborn as sns

            # Optional: allow user to choose features
            all_features = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", 
                            "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
            selected_features = st.multiselect("Select features for KNN", all_features, default=all_features)

            if len(selected_features) == 0:
                st.warning("Please select at least one feature.")
            else:
                X = df_model[selected_features]
                y = (df_model["Delivery_Time_min"] <= 30).astype(int)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Try different k values
                accuracies = []
                k_range = range(1, 21)
                best_k = 1
                best_acc = 0
                best_model = None

                for k in k_range:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)
                    preds = knn.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    accuracies.append(acc)
                    if acc > best_acc:
                        best_k = k
                        best_acc = acc
                        best_model = knn

                st.markdown(f"✅ Best value of k: **{best_k}**")
                st.markdown(f"📈 Best accuracy: **{best_acc:.2%}**")

                # Plot K vs Accuracy
                fig, ax = plt.subplots()
                sns.lineplot(x=list(k_range), y=accuracies, marker="o", ax=ax)
                ax.set_title("K Number × Accuracy")
                ax.set_xlabel("K")
                ax.set_ylabel("Accuracy")
                st.pyplot(fig)



# Page 4: Explainability
elif page == "Explainability 🤔":
    st.title("🤔 Model Explainability with SHAP")

    df_model = df.dropna().copy()
    df_model["Weather"] = LabelEncoder().fit_transform(df_model["Weather"])
    df_model["Traffic_Level"] = LabelEncoder().fit_transform(df_model["Traffic_Level"])
    df_model["Time_of_Day"] = LabelEncoder().fit_transform(df_model["Time_of_Day"])
    df_model["Vehicle_Type"] = LabelEncoder().fit_transform(df_model["Vehicle_Type"])

    features = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", 
                "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
    target = "Delivery_Time_min"

    X = df_model[features]
    y = df_model[target]
    model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.subheader("🌍 Global Feature Importance")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=7, show=False)
    st.pyplot(fig)

    st.subheader("📊 SHAP Summary Plot")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig2)

    st.subheader("🔍 Explain Single Prediction")
    instance = st.slider("Pick a row to explain", 0, len(X)-1, 0)
    fig3, ax3 = plt.subplots()
    shap.plots.waterfall(shap_values[instance], show=False)
    st.pyplot(fig3)

elif page == "Model Tracker 📊":
    st.title("📊 Model Tracker with DagsHub + MLflow")
    st.markdown("This page shows all logged experiments and highlights your best model based on MAE.")

    # 🔧 Set MLflow URI (DagsHub)
    mlflow.set_tracking_uri("https://dagshub.com/zy2869/my-first-repo.mlflow")

    client = MlflowClient()

    # 🔍 Show all experiments so user knows what's available
    experiments = mlflow.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_exp_name = st.selectbox("Choose experiment", experiment_names)

    selected_exp = client.get_experiment_by_name(selected_exp_name)
    runs = client.search_runs(experiment_ids=[selected_exp.experiment_id], order_by=["metrics.MAE ASC"])

    # 📊 Create table
    data = []
    for r in runs:
        data.append({
            "Run ID": r.info.run_id,
            "Model": r.data.tags.get("mlflow.runName", "Unnamed"),
            "MAE": r.data.metrics.get("MAE", None),
            "MSE": r.data.metrics.get("MSE", None),
            "MAPE": r.data.metrics.get("MAPE", None),
        })
    df_runs = pd.DataFrame(data)

    # 🏆 Show sorted models
    st.subheader("Top Performing Models (Sorted by MAE)")
    if not df_runs.empty:
        st.dataframe(df_runs.sort_values("MAE", na_position='last').reset_index(drop=True))
    else:
        st.warning("No runs with MAE metric found in this experiment.")

if page == "Conclusion 📌":
    st.title("📌 Conclusion and Insights")

    st.subheader("🍔 Delivery Strategy Recommendations Based on Our Analysis")

    st.markdown("""
    **Based on our overall analysis**, we found that delivery time is most strongly influenced by a few key operational features: **distance**, **preparation time**, and **traffic level**. These factors consistently showed high predictive value across models and SHAP explanations.

    📍 For instance, our SHAP analysis confirmed that **Distance (km)** had the highest impact on delivery time predictions, while **Preparation Time** also played a major role. When these two were both high, delivery times significantly increased.

    🏍️ Among the different vehicle types, **bikes** were the most frequently used (51%), but they also had more variation in delivery speed depending on other conditions like traffic.

    📈 As distance increases, average delivery time predictably rises—a trend confirmed by both bar charts and regression models.
    """)

    st.subheader("🧠 Key Learnings from Model Comparison")

    st.markdown("""
    - **Linear Regression** offered a strong baseline with an R² of **0.775**.
    - **Decision Trees** gave better interpretability with strong accuracy (~91.5%) but a lower F1-score.
    - **K-Nearest Neighbors (KNN)** with selected features reached **96.05% accuracy**.

    🔍 Our model tracker (with MLflow + DagsHub) revealed that **Huber Regressor** performed best in terms of MAE, making it a great option when minimizing large errors.
    """)

    st.subheader("🚚 Real-World Use Case")

    st.markdown("""
    These results suggest that food delivery platforms could:
    - ✅ Use real-time **distance and traffic** data to adjust estimated delivery windows.
    - ✅ Improve ETAs by accounting for **preparation time** at the vendor.
    - ✅ Recommend **vehicle-type optimizations** during peak or off-peak hours.

    This could lead to improved customer satisfaction, fewer complaints, and better delivery routing decisions.
    """)

    st.subheader("🔧 Future Improvements?")

    st.markdown("""
    1. **Live Traffic API Integration**: Use real-time traffic feeds (e.g., Google Maps API) for more dynamic predictions.
    2. **User Behavior Modeling**: Include customer behavior (e.g., reorder rate, tip likelihood) to improve prioritization.
    3. **Expand Dataset**: Include orders from multiple cities to improve generalization across delivery environments.
    """)

if page == "What-If Simulator 🔁":
    st.title("🔁 What-If Simulator")
    st.markdown("### Adjust inputs to simulate delivery time!")

    df_model = df.dropna().copy()
    df_model["Weather"] = LabelEncoder().fit_transform(df_model["Weather"])
    df_model["Traffic_Level"] = LabelEncoder().fit_transform(df_model["Traffic_Level"])
    df_model["Time_of_Day"] = LabelEncoder().fit_transform(df_model["Time_of_Day"])
    df_model["Vehicle_Type"] = LabelEncoder().fit_transform(df_model["Vehicle_Type"])

    features = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", 
                "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]

    # Train simple model
    X = df_model[features]
    y = df_model["Delivery_Time_min"]
    model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X, y)

    # Input widgets
    st.markdown("#### Input Simulation Variables")
    col1, col2 = st.columns(2)

    with col1:
        distance = st.slider("Distance (km)", 0.5, 25.0, 5.0)
        prep_time = st.slider("Preparation Time (min)", 5, 40, 15)
        experience = st.slider("Courier Experience (yrs)", 0, 10, 2)

    with col2:
        weather = st.selectbox("Weather", df["Weather"].unique())
        traffic = st.selectbox("Traffic Level", df["Traffic_Level"].unique())
        time_of_day = st.selectbox("Time of Day", df["Time_of_Day"].unique())
        vehicle = st.selectbox("Vehicle Type", df["Vehicle_Type"].unique())

    # Encoding user input
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Weather": [LabelEncoder().fit(df["Weather"]).transform([weather])[0]],
        "Traffic_Level": [LabelEncoder().fit(df["Traffic_Level"]).transform([traffic])[0]],
        "Time_of_Day": [LabelEncoder().fit(df["Time_of_Day"]).transform([time_of_day])[0]],
        "Vehicle_Type": [LabelEncoder().fit(df["Vehicle_Type"]).transform([vehicle])[0]],
        "Preparation_Time_min": [prep_time],
        "Courier_Experience_yrs": [experience]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"📦 Estimated Delivery Time: {prediction:.2f} minutes")

    st.caption("⚡ Tip: Try extreme values to simulate peak vs. off-peak hours!")
