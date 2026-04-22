import streamlit as st
from PIL import Image
from src.models.predict import predict

st.title("Invoice Reader (Donut)")

uploaded_file = st.file_uploader("Upload Invoice", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    st.write("Prediction:")
    predict(uploaded_file)
