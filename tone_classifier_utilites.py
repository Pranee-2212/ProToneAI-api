import pickle
import string

def initailize_model_and_vector():
    tone_mapping = {'Aggressive/Frustrated': 0, 'Passive/Ambiguous': 1, 'Professional/Constructive': 2}

# Load the trained model
    with open('./model/tone_classifier_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

# Load the TF-IDF vectorizer
    with open('./model/tfidf_vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    tone_mapping_reverse = {v: k for k, v in tone_mapping.items()}
    return loaded_model,loaded_vectorizer,tone_mapping_reverse

# Sample text messages for testing, covering all three tone classes
sample_messages = {
    "Aggressive/Frustrated (High Confidence)": "This is unacceptable! Fix it immediately.",
    "Aggressive/Frustrated (Low Confidence)": "I'm a little annoyed by this issue.", # Example of a less aggressive message
    "Passive/Ambiguous (High Confidence)": "I guess maybe we could perhaps look at this issue sometime?", # Example with more ambiguity
    "Passive/Ambiguous (Low Confidence)": "Maybe we could look into this issue sometime?",
    "Professional/Constructive (High Confidence)": "Could you please provide a detailed update on the status of the project?", # Example with more detail
    "Professional/Constructive (Low Confidence)": "Could you please provide an update on the status?",
    "Aggressive/Frustrated (Another Example)": "Why is this ticket still open? You need to fix it now!",
    "Passive/Ambiguous (Another Example)": "It's not a big deal, but the client mentioned something.",
    "Professional/Constructive (Another Example)": "Thank you for your work. What immediate steps can be taken?"

}

# We need the tone_mapping dictionary from the previous steps



def predict_result(sample_message):
       # Preprocess the sample message (lowercase and remove punctuation)
    loaded_model, loaded_vectorizer, tone_mapping_reverse = initailize_model_and_vector()
    processed_message = sample_message.lower().translate(str.maketrans('', '', string.punctuation))

    # Vectorize the processed message using the loaded vectorizer
    vectorized_message = loaded_vectorizer.transform([processed_message])

    # Predict the tone class and get prediction probabilities
    predicted_label = loaded_model.predict(vectorized_message)[0]
    prediction_proba = loaded_model.predict_proba(vectorized_message)[0]

    # Map the numerical label back to the tone class
    predicted_tone_class = tone_mapping_reverse[predicted_label]

    # Get the confidence level for the predicted class
    confidence_level = prediction_proba[predicted_label]

    # Initialize action_required to False
    action_required = False

    # Add conditions for action_required
    if predicted_tone_class == 'Aggressive/Frustrated':
        action_required = True
    elif predicted_tone_class == 'Passive/Ambiguous' and confidence_level < 0.512:
        action_required = True
     
    # Print the results
    
    return {
        'Predicted_Tone_Class': predicted_tone_class,
        'Confidence_Level': confidence_level,
        'Action_Required': action_required
    }

# Iterate through sample messages and predict tone class