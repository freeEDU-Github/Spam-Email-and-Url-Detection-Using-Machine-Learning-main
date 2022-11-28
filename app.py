import sys
import pandas as pd
import numpy as np
import glob
import sklearn
import nltk
from PIL import Image

nltk.download('stopwords')
import tensorflow as tf
import pickle
import joblib
import streamlit as st

mymodel = joblib.load(open('spam_detector_model.pkl','rb'))

vectorizer = joblib.load(open('vectorizer.pickle','rb'))


def main():
    st.title("Spam Email Analysis with a Machine Learning Approach")

    st.subheader("Spam Email Analysis")
    st.markdown("Spam email is unsolicited and unwanted junk email that is sent in bulk to a random recipient list. Spam is typically sent for commercial purposes. Botnets, or networks of infected computers, can send it in massive quantities.")
    image = Image.open('spam.png')
    st.image(image, caption='Spam Email Detector')

    st.subheader(
        "The primary goal of this project is to identify whether the email messages are spam or not")

    sample_dataset = pd.read_csv("data-set/spam.csv")
    st.dataframe(sample_dataset)

    input_val = st.text_input("Give me some email messages to work on : ")


    if st.button("Predict"):
        prediction = mymodel.predict(vectorizer.transform([input_val]))

        if prediction == 0:
            st.success("This is not a spam")

        elif prediction == 1:
            st.error("🚨️ALERT -  THIS IS A SPAM! 🚨")


if __name__ == '__main__':
    main()