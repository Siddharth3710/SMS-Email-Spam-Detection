import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenization

    y = [i for i in text if i.isalnum()]  # Remove special characters

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords & punctuation

    y = [ps.stem(i) for i in y]  # Stemming

    return " ".join(y)

# Load models (Use raw string or double backslashes)
tfidf = pickle.load(open(r'C:\Users\DeLL\OneDrive\Desktop\College Projects\kaggle\sms spam detector\vectorizer.pkl', 'rb'))
model1 = pickle.load(open("C:/Users/DeLL/OneDrive/Desktop/College Projects/kaggle/sms spam detector/model (1).pkl", "rb"))


# Streamlit UI
st.title('SMS Spam Detector')

input_sms = st.text_area("Enter the message you want to check")

if st.button('Predict'):
    # Preprocess text
    transformed_sms = transform_text(input_sms)

    # Vectorize text
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model1.predict(vector_input)

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
