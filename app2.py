import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import base64  # For image encoding

# Function to convert image to Base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Load the Excel file and extract relevant values
excel_data = pd.ExcelFile('NEURO_ExampleMacro.xlsx')
precr_cop_calculations_df = excel_data.parse('PreCR_Cop_Calculations')

# Extract beta and intercept values
beta_craving = precr_cop_calculations_df.loc[precr_cop_calculations_df['Unnamed: 0'] == 'Pre Cue Craving', 'β'].values[0]
beta_coping = precr_cop_calculations_df.loc[precr_cop_calculations_df['Unnamed: 0'] == 'Coping', 'β'].values[0]
intercept = precr_cop_calculations_df.loc[precr_cop_calculations_df['Unnamed: 0'] == 'Constant/Intercept', 'β*score'].values[0]

# Fixed threshold
threshold = 0.4649749

# Define the classification function
def classify_aud(pre_cue_craving_score, coping_score):
    log_odds = intercept + (beta_craving * pre_cue_craving_score) + (beta_coping * coping_score)
    probability = 1 / (1 + np.exp(-log_odds))
    classification = "AUD" if probability >= threshold else "NON-AUD"
    return probability, classification

# Streamlit Web Application
st.title("AUD Classification Tool")

# Beverage desirability images (paths to uploaded files)
image_paths = [
    "water28 YES.bmp", "water23 YES.bmp", "water25 YES.bmp",
    "water30 YES.bmp", "water24 YES.bmp", "water2 YES.bmp",
    "water16 YES.bmp", "water19 YES.bmp", "water15 YES.bmp",
    "water18 YES.bmp", "water12 YES.bmp", "water13 YES.bmp"
]

st.write("**Beverage Desirability Questions**")
desirability_scores = []
for idx, image_path in enumerate(image_paths):
    if os.path.exists(image_path):
        st.write(f"How desirable is this beverage? (1 = Minimum, 10 = Maximum)")
        # Center the image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{get_image_base64(image_path)}" width="200" />
            </div>
            """,
            unsafe_allow_html=True
        )
        score = st.slider(f"Desirability score for Beverage {idx+1}:", 1, 10, 1)
        desirability_scores.append(score)
    else:
        st.warning(f"Image {idx+1} not found at {image_path}")

# Ask craving question
st.write("**Craving Questions**")
a = st.slider("a. On a scale of 1-10, How much do you want an alcoholic drink right now?", 1, 10, 1)
b = st.slider("b. On a scale of 1-10, How much do you crave an alcoholic drink right now?", 1, 10, 1)
c = st.slider("c. On a scale of 1-10, How much do you desire an alcoholic drink right now?", 1, 10, 1)
d = st.slider("d. On a scale of 1-10, How high is your urge for an alcoholic drink right now?", 1, 10, 1)
pre_cue_craving_score = (a + b + c + d)/4


# Define labels for coping questions
scale_labels = {
    0: "Never/Almost never",
    1: "Some of the time",
    2: "Half of the time",
    3: "Most of the time",
    4: "Almost always/Always"
}

# Ask coping questions and calculate total coping score
st.write("**Coping Questions**")
q1 = st.radio("1. I drink to relax.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q2 = st.radio("2. I drink to forget my worries.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q3 = st.radio("3. I drink to feel more self-confident or sure of myself.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q4 = st.radio("4. I drink because it helps when I feel depressed or nervous.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q5 = st.radio("5. I drink to cheer up when I am in a bad mood.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])

# Calculate the total coping score
coping_score = q1 + q2 + q3 + q4 + q5

# Display total scores
st.write(f"Your total coping score is: **{coping_score}**")
st.write(f"Your craving score is: **{pre_cue_craving_score}**")

# Calculate probability and classification on button click
if st.button("Classify"):
    probability, classification = classify_aud(pre_cue_craving_score, coping_score)
    
    # Display the probability and classification
    st.write(f"Model's Predicted Score: **{probability:.3f}** (Threshold: **{threshold:.3f}**)")

    # Display the classification result with color and large font
    color = "red" if classification == "AUD" else "green"
    st.markdown(f"<span style='color:{color}; font-size:32px; font-weight:bold;'>Classification: {classification}</span>", unsafe_allow_html=True)
    
    # Gauge chart visualization
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=probability,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold], 'color': "lightgreen"},
                {'range': [threshold, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(title="Model's Predicted Score Gauge for AUD Classification")
    st.plotly_chart(fig)

st.write("---")
st.write("**Note:** This classification was done using binary logistic regression. Accuracy is 89%. For more information refer to: <LINK to the PAPER>")

