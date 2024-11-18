import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# Ask the user if they have the craving and coping scores
user_response = st.radio(
    "Do you have the coping and craving scores?",
    ("Yes", "No, I would like to be asked the questions.")
)

if user_response == "Yes":
    # User provides craving and coping scores
    pre_cue_craving_score = st.number_input("Enter Pre Cue Craving Score:", min_value=0.0)
    coping_score = st.number_input("Enter Coping Score:", min_value=0.0)
else:
    # Ask craving question
    pre_cue_craving_score = st.slider("On a scale of 1-10, how much do you crave alcohol right now?", 1, 10, 1)
    
    # Define the labels for the scale
scale_labels = {
    0: "Never",
    1: "Almost never",
    2:"Some of the time",
    3: "Half of the time",
    4: "Most of the time",
    5: "Almost always/Always"
}

# Ask coping questions and calculate total coping score
st.write("Rate the following statements on a scale of 0-4:")
st.write("0 = Never, 1 = Almost never, 2 = Some of the time, 3 = Half of the time, 4 = Most of the time, 5 = Almost always/Always:")

q1 = st.radio("1. I drink to relax.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q2 = st.radio("2. I drink to forget my worries.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q3 = st.radio("3. I drink to feel more self-confident or sure of myself.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q4 = st.radio("4. I drink because it helps when I feel depressed or nervous.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])
q5 = st.radio("5. I drink to cheer up when I am in a bad mood.", list(scale_labels.keys()), format_func=lambda x: scale_labels[x])

# Calculate the total coping score
coping_score = q1 + q2 + q3 + q4 + q5

# Display the total coping score
st.write(f"Your total coping score is: **{coping_score}**")
    # Display the total coping score and craving score
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
    
    # Gauge chart visualization without number in the center
    fig = go.Figure(go.Indicator(
        mode="gauge",  # Only gauge, no number or delta
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
    fig.update_layout(
        title="Model's Predicted Score Gauge for AUD Classification",
        annotations=[
            {
                "x": 0.5, "y": -0.2, "xref": "paper", "yref": "paper",
                "text": f"Classification: {classification}",
                "showarrow": False,
                "font": {"size": 24, "color": color}
            }
        ]
    )
    st.plotly_chart(fig)
# Add a note about the model and accuracy
st.write("---")
st.write("**Note:** This classification was done using a logistic net regression and receiver operating characteristic analysis, with an accuracy of 89%. As a result, the model's predicted score does not necessarily reflect the real-life probability of AUD.")

