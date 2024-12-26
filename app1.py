import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('artifacts/preprocessor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a function for making predictions
def make_prediction(input_data):
    # Preprocess the input data in the same way it was done during training
    label_encoder = LabelEncoder()

    # Encoding categorical variables (Ensure these columns were encoded during training)
    input_data['gender'] = label_encoder.fit_transform(input_data['gender'])
    input_data['race_ethnicity'] = label_encoder.fit_transform(input_data['race_ethnicity'])
    input_data['parental_level_of_education'] = label_encoder.fit_transform(input_data['parental_level_of_education'])
    input_data['lunch'] = label_encoder.fit_transform(input_data['lunch'])
    input_data['test_preparation_course'] = label_encoder.fit_transform(input_data['test_preparation_course'])

    # Ensure the input data has the correct columns and order for the model
    expected_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 
                        'test_preparation_course', 'reading_score', 'writing_score']
    input_data = input_data[expected_columns]

    # Model prediction
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title('Math Score Prediction')

st.write("""
This application uses the trained model to predict the math score based on user input features.
""")

# Input fields for user to provide data
gender = st.selectbox('Gender', ['female', 'male'])
race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D'])
parental_level_of_education = st.selectbox('Parental Level of Education', ["some college", "associate's degree", "high school", "bachelor's degree", "master's degree", "some high school"])
lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
test_preparation_course = st.selectbox('Test Preparation Course', ['none', 'completed'])

# You can add more input fields if needed
reading_score = st.number_input('Reading Score', min_value=0, max_value=100)
writing_score = st.number_input('Writing Score', min_value=0, max_value=100)

# Collect all inputs into a dataframe
input_data = pd.DataFrame({
    'gender': [gender],
    'race_ethnicity': [race_ethnicity],
    'parental_level_of_education': [parental_level_of_education],
    'lunch': [lunch],
    'test_preparation_course': [test_preparation_course],
    'reading_score': [reading_score],
    'writing_score': [writing_score]
})

# Display a button to make the prediction
if st.button('Predict'):
    prediction = make_prediction(input_data)
    st.write(f"Predicted Math Score: {prediction[0]}")
