import streamlit as st
import requests


st.set_page_config(page_title="Streamlit App", page_icon="🧪", layout="centered")
st.title("Streamlit is running")

url = st.text_input("Test a URL with requests", "https://httpbin.org/get")
if st.button("Fetch"):
    try:
        resp = requests.get(url, timeout=10)
        st.write("Status:", resp.status_code)
        st.json(resp.json())
    except Exception as e:
        st.error(str(e))

