from os import path
import streamlit as st

# import pickle

# from tensorflow import keras
import tensorflow as tf
import torch
from torch import nn
from transformers import BertModel, BertTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-cased"
MODEL_PATH = path.join(path.dirname(__file__), "bert_model.h5")


# Build the Sentiment Classifier class
class SentimentClassifier(nn.Module):
    # Constructor class
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


@st.cache_resource
def load_model_and_tokenizer():
    model = SentimentClassifier(3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model, BertTokenizer.from_pretrained("bert-base-cased")


def predict(content):
    model, tokenizer = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoded_review = tokenizer.encode_plus(
        content,
        max_length=160,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded_review["input_ids"].to(device)
    attention_mask = encoded_review["attention_mask"].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    class_names = ["negative", "neutral", "positive"]

    return class_names[prediction]


def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")

    st.title("Seminar Công nghệ Tri thức - Transformer trong NLP")
    st.markdown(
        """
        **Team members:**
        | Student ID | Full Name                |
        | ---------- | ------------------------ |
        | 19120600   | Bùi Nguyên Nghĩa         |
        | 19120607   | Phạm Thị Nguyệt          |
        """
    )
    
    # giving a title to our page
    st.title("Sentiment analysis")
    contents = st.text_area(
        "Please enter reviews/sentiment/setences/contents:",
        placeholder="Enter your text here",
        height=200,
    )

    prediction = ""

    # Create a prediction button
    if st.button("Analyze Sentiment"):
        stripped = contents.strip()
        if not stripped:
            st.error("Please enter some text.")
            return

        prediction = predict(contents)
        if prediction == "positive":
            st.success("This is positive 😄")
        elif prediction == "negative":
            st.error("This is negative 😟")
        else:
            st.warning("This is neutral 🙂")

    upload_file = st.file_uploader("Or upload a file", type=["txt"])
    if upload_file is not None:
        contents = upload_file.read().decode("utf-8")

        for line in contents.splitlines():
            line = line.strip()
            if not line:
                continue

            prediction = predict(line)
            if prediction == "positive":
                st.success(line + "\n\nThis is positive 😄")
            elif prediction == "negative":
                st.error(line + "\n\nThis is negative 😟")
            else:
                st.warning(line + "\n\nThis is neutral 🙂")


if __name__ == "__main__":
    main()
