# import streamlit as st
# from transformers import BertForSequenceClassification, AutoTokenizer, pipeline

# import torch


# # Load the sentiment model and tokenizer
# @st.cache_resource
# def load_sentiment_model_and_tokenizer():
#     model_save_path = r".\saved_model"
#     model = BertForSequenceClassification.from_pretrained(model_save_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_save_path)
#     return model, tokenizer

# # Load the emotion classifier pipeline
# @st.cache_resource
# def load_emotion_classifier():
#     return pipeline(
#         "text-classification",
#         model='bhadresh-savani/distilbert-base-uncased-emotion',
#         return_all_scores=True,
#         truncation=True,
#         max_length=512
#     )

# # Initialize models and tokenizer
# sentiment_model, sentiment_tokenizer = load_sentiment_model_and_tokenizer()
# emotion_classifier = load_emotion_classifier()

# def predict_sentiment(review):
#     """Predict sentiment using the pre-trained BERT model."""
#     inputs = sentiment_tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
#     outputs = sentiment_model(**inputs)
#     sentiment = torch.argmax(outputs.logits, dim=1).item()
#     return "Positive" if sentiment == 1 else "Negative"

# def predict_emotions(review):
#     """Predict emotions using the emotion classifier pipeline."""
#     return emotion_classifier(review)

# # Streamlit App Layout
# st.title("Sentiment and Emotion Analysis")
# st.write("Enter a review to analyze its sentiment and emotional tone.")

# # Input text box
# review = st.text_area("Review Text", placeholder="Type your review here...")

# if st.button("Analyze"):
#     if review.strip():
#         # Predict sentiment
#         sentiment = predict_sentiment(review)
#         st.write(f"**Sentiment:** {sentiment}")

#         # Predict emotions
#         emotion_scores = predict_emotions(review)
#         st.write("**Emotion Scores:**")
#         for emotion in emotion_scores[0]:
#             st.write(f"{emotion['label']}: {emotion['score']:.2f}")
#     else:
#         st.warning("Please enter a review to analyze.")
import streamlit as st 
from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
import torch
import openai

# Set OpenAI API key
openai.api_key='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'




# Load the sentiment model and tokenizer
@st.cache_resource
def load_sentiment_model_and_tokenizer():
    model_save_path = r"saved_model"
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    return model, tokenizer

# Load the emotion classifier pipeline
@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model='bhadresh-savani/distilbert-base-uncased-emotion',
        return_all_scores=True,
        truncation=True,
        max_length=512
    )

# Initialize models and tokenizer
sentiment_model, sentiment_tokenizer = load_sentiment_model_and_tokenizer()
emotion_classifier = load_emotion_classifier()

def predict_sentiment(review):
    """Predict sentiment using the pre-trained BERT model."""
    inputs = sentiment_tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if sentiment == 1 else "Negative"

def predict_emotions(review):
    """Predict emotions using the emotion classifier pipeline."""
    return emotion_classifier(review)

def chatbot_response(query):
    """Generate a chatbot response for application-related queries."""
    prompt = f"Provide a generalized answer to the following application-related query:\n{query}\nAvoid unnecessary or irrelevant details."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def refine_review(review):
    """Refine a review into natural and polished language."""
    prompt = f"Refine the following customer review to make it sound more natural and professional:\n{review}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit App Layout
st.title("Sentiment and Emotion Analysis with Chatbot and Review Refinement")
st.write("Enter a review to analyze its sentiment and emotional tone, or ask a query about the application.")

# Input text box for review analysis
review = st.text_area("Review Text", placeholder="Type your review here...")

# Analyze button
if st.button("Analyze"):
    if review.strip():
        # Predict sentiment
        sentiment = predict_sentiment(review)
        st.write(f"**Sentiment:** {sentiment}")

        # Predict emotions
        emotion_scores = predict_emotions(review)
        st.write("**Emotion Scores:**")
        for emotion in emotion_scores[0]:
            st.write(f"{emotion['label']}: {emotion['score']:.2f}")
    else:
        st.warning("Please enter a review to analyze.")

# Refine review button
if st.button("Refine Review"):
    if review.strip():
        refined_review = refine_review(review)
        st.write("**Refined Review:**")
        st.write(refined_review)
    else:
        st.warning("Please enter a review to refine.")

# Chatbot input
query = st.text_input("Ask the Chatbot", placeholder="Type your query here...")

if st.button("Get Response"):
    if query.strip():
        response = chatbot_response(query)
        st.write("**Chatbot Response:**")
        st.write(response)
    else:
        st.warning("Please enter a query for the chatbot.")
