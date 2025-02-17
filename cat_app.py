import streamlit as st
import joblib
import pandas as pd

def load_model(model_path):
    """Load the trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_category(model, question):
    """Predict category for a single question"""
    try:
        # Create a single-row DataFrame
        df = pd.DataFrame([question], columns=['Question'])
        prediction = model.predict(df['Question'])[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Set up the Streamlit app
st.title('Quiz Question Category Predictor')
st.write('Enter a quiz question to predict its category')

# Load the model at app startup
model = load_model('categorymodel.pkl')

if model:
    # Create input text area for the question
    question = st.text_area('Enter your quiz question:', height=100)
    
    if st.button('Predict Category'):
        if question.strip():
            # Make prediction
            category = predict_category(model, question)
            if category:
                st.success(f'Predicted Category: {category}')
        else:
            st.warning('Please enter a question first.')
else:
    st.error('Failed to load the model. Please ensure the model file exists and is accessible.')