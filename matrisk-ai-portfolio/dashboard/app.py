import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

# -----------------------------
# Load Dataset
# -----------------------------

material_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "material_features.csv"
)

df = pd.read_csv(material_path)

# -----------------------------
# Load Trained Model
# -----------------------------

model_path = (
    BASE_DIR
    / "models"
    / "trained"
    / "random_forest_bulk_modulus.pkl"
)

model = joblib.load(model_path)

# -----------------------------
# Streamlit Config
# -----------------------------

st.set_page_config(
    page_title="MatRisk AI Dashboard",
    layout="wide"
)

st.title("MatRisk AI Dashboard")
st.subheader(
    "Material Property & Infrastructure Risk Intelligence"
)

# -----------------------------
# Dataset Overview
# -----------------------------

st.header("Dataset Overview")

st.write(f"Dataset Shape: {df.shape}")

st.dataframe(df.head())

# -----------------------------
# Distribution Plot
# -----------------------------

st.header("Bulk Modulus Distribution")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df["bulk_modulus_GPa"], bins=30)

ax.set_xlabel("Bulk Modulus")
ax.set_ylabel("Frequency")
ax.set_title("Bulk Modulus Distribution")

st.pyplot(fig)

# -----------------------------
# Prediction Section
# -----------------------------

st.header("Bulk Modulus Prediction")

feature_cols = list(model.feature_names_in_)

input_data = {}

for col in feature_cols:
    input_data[col] = float(df[col].mean())

st.info(
    "Using dataset mean values for baseline prediction."
)

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_cols]

prediction = model.predict(input_df)[0]

st.subheader("Predicted Bulk Modulus")

st.success(f"{prediction:.4f}")

# -----------------------------
# Feature Importance
# -----------------------------

st.header("Feature Importance")

importance_image = (
    BASE_DIR
    / "reports"
    / "figures"
    / "feature_importance.png"
)

if importance_image.exists():
    st.image(str(importance_image))
else:
    st.warning("Feature importance image not found.")

# -----------------------------
# SHAP Explainability
# -----------------------------

st.header("SHAP Explainability")

shap_image = (
    BASE_DIR
    / "reports"
    / "figures"
    / "shap_bar_bulk_modulus.png"
)

if shap_image.exists():
    st.image(str(shap_image))
else:
    st.warning("SHAP explainability image not found.")

# -----------------------------
# Infrastructure Results
# -----------------------------

st.header("Infrastructure Risk Results")

infra_plot = (
    BASE_DIR
    / "reports"
    / "figures"
    / "infrastructure_actual_vs_predicted.png"
)

if infra_plot.exists():
    st.image(str(infra_plot))
else:
    st.warning("Infrastructure plot not found.")

# -----------------------------
# Commodity Risk Results
# -----------------------------

st.header("Commodity Risk Results")

commodity_plot = (
    BASE_DIR
    / "reports"
    / "figures"
    / "commodity_actual_vs_predicted.png"
)

if commodity_plot.exists():
    st.image(str(commodity_plot))
else:
    st.warning("Commodity prediction plot not found.")

st.success("Day 9 Streamlit Dashboard Loaded Successfully.")