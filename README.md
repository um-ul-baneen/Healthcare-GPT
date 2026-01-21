HEALTHCARE ASSISTANT (STREAMLIT + TRANSFORMERS)

OVERVIEW
The HealthCare Assistant is a Streamlit-based web application that leverages custom transformer models
to generate medical responses. The system allows users to ask healthcare-related questions and receive
AI-generated answers from different professional perspectives, such as General, Surgeon, Cardiologist,
and Pharmacist.

KEY FEATURES
- Interactive Streamlit web interface
- Multiple custom medical AI models:
  * NEENMED – Causal language model for general medical text
  * DECMED – Text-generation pipeline for disease-centric responses
  * NBMED – Sequence-to-sequence medical model
- Perspective-based responses (“Spaces”) to simulate expert viewpoints
- Clean and responsive custom UI styling
- Cached model loading for improved performance
- Sidebar-based navigation (Home, Spaces, About)

APPLICATION PAGES

HOME
- Select AI model (NEENMED / DECMED / NBMED)
- Choose a professional perspective (Spaces)
- Enter a medical query
- Generate and view AI responses in a styled output container

SPACES
- Explains the concept of role-based perspectives
- Allows users to explore how professional context affects responses
- Feature currently under development

ABOUT
- Describes the purpose and scope of the application
- Lists system features
- Credits the creators

TECHNOLOGIES USED
- Python
- Streamlit
- Hugging Face Transformers
- Custom-trained medical language models
- HTML & CSS for UI customization

INSTALLATION & SETUP

1. Clone the Repository
git clone https://github.com/your-username/healthcare-assistant.git
cd healthcare-assistant

2. Install Dependencies
pip install streamlit transformers torch

3. Run the Application
streamlit run app.py

MODEL DETAILS
The application uses the following pretrained models hosted on Hugging Face:
- NamishKhurshid/NEENMED
- NamishKhurshid/DECMED
- NamishKhurshid/NBMED

Models are loaded using Streamlit caching to reduce reload time and improve performance.

INTENDED USE
- Academic learning and demonstrations
- AI in healthcare research projects
- Medical text generation experimentation

DISCLAIMER
This application is for educational and research purposes only.
It does NOT replace professional medical advice, diagnosis, or treatment.

Bachelor of Science in Artificial Intelligence (BSAI)
SZABIST University
