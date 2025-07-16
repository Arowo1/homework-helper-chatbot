import streamlit as st
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgb_model.json')
vectorizer = joblib.load('models/vectorizer.pkl')
encoder = joblib.load('models/label_encoder.pkl')

rnn_model = tf.keras.models.load_model('models/rnn_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

st.title("📚 Homework Helper Chatbot 🤖")
st.caption("Ask me any academic question. I'll guess the subject and help you answer!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask your homework question...")

if user_input:
    vec_input = vectorizer.transform([user_input])
    pred = xgb_model.predict(vec_input)
    subject = encoder.inverse_transform(pred)[0]

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=20)
    prediction = rnn_model.predict(padded)
    word_index = np.argmax(prediction[0])

    next_word = "<unknown>"
    for word, index in tokenizer.word_index.items():
        if index == word_index:
            next_word = word
            break

    ai_response = (
        f"That sounds like a **{subject}** question.\n"
        f"📌 Here's a tip: Think about how this relates to {subject.lower()} concepts.\n"
        f"💡 Suggested word to start your answer: **{next_word}**"
    )

    st.session_state.chat_history.append((user_input, ai_response))

for user_msg, bot_reply in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
