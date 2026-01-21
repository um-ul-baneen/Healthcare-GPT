import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Set page layout
st.set_page_config(page_title="HealthCare Assistant", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        border: none;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #007BFF;
    }
    h2 {
        font-family: 'Arial', sans-serif;
        color: #007BFF;
    }
    h3 {
        font-family: 'Arial', sans-serif;
        color: #0056b3;
    }
    p {
        font-family: 'Arial', sans-serif;
        color: #666666;
    }
    .response-text {
        font-family: 'Arial', sans-serif;
        color: white;
        background-color: #007BFF;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for Index
st.sidebar.title("Index")
page = st.sidebar.radio("Go to:", ["Home", "About", "Spaces"])

# Load models and tokenizers
@st.cache_resource
def load_neenmed_model():
    tokenizer = AutoTokenizer.from_pretrained("NamishKhurshid/NEENMED")
    model = AutoModelForCausalLM.from_pretrained("NamishKhurshid/NEENMED")
    return tokenizer, model

@st.cache_resource
def load_decmed_model():
    return pipeline("text-generation", model="NamishKhurshid/DECMED")

@st.cache_resource
def load_nbmed_model():
    tokenizer = AutoTokenizer.from_pretrained("NamishKhurshid/NBMED")
    model = AutoModelForSeq2SeqLM.from_pretrained("NamishKhurshid/NBMED")
    return tokenizer, model

# Load models
neenmed_tokenizer, neenmed_model = load_neenmed_model()
decmed_pipeline = load_decmed_model()
nbmed_tokenizer, nbmed_model = load_nbmed_model()

# Pages
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>HealthCare Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>At Your Service</p>", unsafe_allow_html=True)

    # User Input and Model Selection
    st.markdown("<h2>Ask a Medical Query</h2>", unsafe_allow_html=True)
    model_selection = st.selectbox("Select Model", ["NEENMED", "DECMED", "NBMED"], index=0)
    perspective_selection = st.selectbox(
        "Choose Spaces", ["General", "Surgeon", "Cardiologist", "Pharmacist"], index=0
    )
    user_input = st.text_input("Type your query below:", placeholder="Enter a medical query here", key="user_input")
    submit_button = st.button("Generate Response")

    # Generate response
    if submit_button and user_input.strip():
        with st.spinner("Generating response..."):
            perspective_prefix = f"[{perspective_selection}] " if perspective_selection != "General" else ""
            user_query = perspective_prefix + user_input

            if model_selection == "NEENMED":
                inputs = neenmed_tokenizer.encode(user_query, return_tensors="pt")
                outputs = neenmed_model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True)
                response = neenmed_tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif model_selection == "DECMED":
                response = decmed_pipeline(user_query, max_length=100, num_return_sequences=1, do_sample=True)[0]['generated_text']
            else:
                inputs = nbmed_tokenizer.encode(user_query, return_tensors="pt")
                outputs = nbmed_model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True)
                response = nbmed_tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("<h2>Response</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='response-text'>{response}</p>", unsafe_allow_html=True)

elif page == "Spaces":
    st.markdown("<h1 style='text-align: center;'>Explore Spaces</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <h2>What are Spaces?</h2>
        <p>
        Spaces typically refer to creating context-specific environments or perspectives that influence how the model responds to queries. It allows customization of the assistant's tone, expertise, or behavior based on predefined settings or prompts and questions from different professional perspectives,
        such as Surgeons, Cardiologists, or Pharmacists. Choose a perspective and ask specific queries to get
        tailored answers based on the role you select.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h2>Try Asking a Question!</h2>", unsafe_allow_html=True)

    perspective_selection = st.selectbox(
        "Choose Spaces", ["General", "Surgeon", "Cardiologist", "Pharmacist"], index=0
    )
    user_input = st.text_input("Enter your question below:", placeholder="Ask about a healthcare topic.")
    submit_button = st.button("Get Answer")

    if submit_button and user_input.strip():
        st.markdown(f"<p>Generating a response from the perspective of a <b>{perspective_selection}</b>...</p>", unsafe_allow_html=True)
        st.markdown("<p>Feature still in development!</p>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("<h1 style='text-align: center;'>About This Assistant</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <h2>Purpose</h2>
        <p>
        This application uses custom GPT models to generate medical text. It aims to assist healthcare professionals
        and individuals by providing insights into diseases, predicting potential diagnoses, and offering proactive
        suggestions.
        </p>
       
        <h2>Features</h2>
        <ul>
            <li>Generates disease-specific information using custom AI models.</li>
            <li>Supports multiple AI models for generating medical text: NEENMED, DECMED, and NBMED.</li>
            <li>Allows generating responses from different perspectives such as Surgeons, Cardiologists, and Pharmacists.</li>
        </ul>
       
        <h2>About the Creators</h2>
        <p>
        This application was created by <b>Namish Binte Khurshid</b> and <b>Um-Ul-Baneen</b>, students of the Bachelor of Science
        in Artificial Intelligence (BSAI) program at SZABIST. They designed this system to assist healthcare
        professionals and individuals seeking medical knowledge.
        </p>
        """,
        unsafe_allow_html=True,
    )
