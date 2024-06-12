import streamlit as st
from groquesco import get_response

st.title("Virtual Shopping Assistant")

user_question = st.text_input("quero uma celular azul com 8gb de ram")

if user_question:
    response = get_response(user_question)
    st.write(response,unsafe_allow_html=True)
