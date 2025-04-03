import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st

# loading the model
model = tf.keras.models.load_model("model.h5")

# loading the encoders and scalers

with open("label_encoder_gender.pkl","rb") as file:
    LB = pickle.load(file)

with open("column_transformer_geo.pkl","rb") as file:
    CT = pickle.load(file)

with open("standard_scaler.pkl","rb") as file:
    SC = pickle.load(file)

##streamlitapp
st.title("Customer Churn Prediction")

# Get categories from ColumnTransformer (Geography column)
geography_categories = CT.named_transformers_["OHE"].categories_[0]

# Get categories from LabelEncoder (Gender)
gender_categories = LB.classes_

# Streamlit UI
st.title("ANN Classification App")
st.write("Enter customer details below:")

# User Input Fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", geography_categories)  # Dynamic Categories from OneHotEncoder
gender = st.selectbox("Gender", gender_categories)  # Dynamic Categories from LabelEncoder
age = st.number_input("Age", min_value=18, max_value=100, value=40)

# Sliders for Tenure and Number of Products
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
num_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)

balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
has_credit_card = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_credit_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

input_data["Gender"] = LB.transform(input_data["Gender"])
encoded_input = CT.transform(input_data)
encoded_df = pd.DataFrame(encoded_input, columns=CT.get_feature_names_out())
input_scaled = SC.transform(encoded_df)

prediction = model.predict(input_scaled)
prediction_Prob = prediction[0][0]

st.write(f"Churn probability: {prediction_Prob:.2f}")

if prediction_Prob > 0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")