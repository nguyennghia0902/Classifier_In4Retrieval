import streamlit as st
from st_pages import Page, show_pages

st.set_page_config(page_title="Information Retrieval", page_icon="🏠")

show_pages(
    [
        Page("streamlit_app.py/Homepage.py", "Home", "🏠"),
        Page(
            "streamlit_app.py/pages/in4Retrieval.py", "Information Retrieval", "📝"
        ),
    ]
)

st.title("Text Mining & Applications - 2021_22")
st.markdown(
    """
    **Team members:**
    | Student ID | Full Name                |
    | ---------- | ------------------------ |
    | 19120600   | Bùi Nguyên Nghĩa         |
    """
)

st.header("Technology used")
st.markdown(
    """
    In this demo, we used BERT as the model for sentiment analysis. BERT is a transformer-based model that was proposed in 2018 by Google.
    It is a pre-trained model that can be used for various NLP tasks such as sentiment analysis, question answering, etc.
    """
)


