import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/newspaper-background-concept_23-2149501641.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 3rem;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(27, 67, 50, 0.9);
        padding: 1rem 2rem;
        border-radius: 8px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    .nav-links a {
        color: #FFFFFF !important;
        text-decoration: none !important;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .nav-links a:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFFFFF !important;
    }
    .hero {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
        margin-bottom: 4rem;
    }
    .hero-text h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #1B4332;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8);
    }
    .hero-text p {
        font-size: 1.25rem;
        color: #2D6A4F;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .btn, .btn:hover, .btn:active, .btn:focus {
        color: #FFFFFF !important;
        text-decoration: none !important;
    }
    .btn {
        background: linear-gradient(135deg, #95D5B2 0%, #2D6A4F 100%);
        padding: 0.9rem 2rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%);
    }
    .btn:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .btn::after {
        content: '→';
        margin-left: 8px;
        transition: transform 0.3s ease;
    }
    .btn:hover::after {
        transform: translateX(4px);
    }
    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin: 4rem 0;
        text-align: center;
    }
    .feature {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .feature i {
        font-size: 2rem;
        color: #2D6A4F;
        margin-bottom: 1rem;
    }
    .analysis-section {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    .result {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 0.5rem;
        display: none;
    }
    .fake {
        background-color: #FEE2E2;
        border-left: 4px solid #DC2626;
    }
    .real {
        background-color: #DCFCE7;
        border-left: 4px solid #22C55E;
    }
    .confidence {
        font-weight: bold;
        margin: 1rem 0;
    }
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
    }
    .badge-fake {
        background-color: #D8F3DC;
        color: #1B4332;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #95D5B2;
    }
    .badge-real {
        background-color: #22C55E;
    }
    .why-link {
        display: block;
        text-align: right;
        margin-top: 1rem;
        color: #2D6A4F;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    .why-link:hover {
        color: #1B4332;
    }
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
        border-top: 1px solid #D8F3DC;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_components():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_components()

# Header
st.markdown("""
<div class="header">
    <h1>Fake News Detector</h1>
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#how-it-works">How it works</a>
        <a href="#">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-text">
        <h1>Detect Fake News Instantly</h1>
        <p>Reliable fake news detection powered by AI.</p>
        <a href="#analyze" class="btn" style="width: fit-content; padding: 0.9rem 2.5rem;">Test News Now</a>
    </div>
    <div class="hero-image">
        <img src="https://news.ubc.ca/wp-content/uploads/2025/04/adobestock_1017659063.jpeg" 
             alt="Fake to Fact" 
             style="max-width: 100%; height: auto; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
    </div>
</div>
""", unsafe_allow_html=True)

# How It Works Section
st.markdown("## How It Works", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%; border: 1px solid #D8F3DC; transition: all 0.3s ease;">
        <div style="background: #D8F3DC; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; transition: transform 0.3s ease;">
        <i class="fas fa-file-alt" style="font-size: 1.5rem; color: #1B4332;"></i>
        </div>
        <h3 style="color: #1F2937; font-size: 1.1rem; text-align: center; margin-bottom: 0.75rem;">1. Input Article</h3>
        <p style="color: #4B5563; font-size: 0.9rem; text-align: center; line-height: 1.5; margin: 0;">Paste the news article or upload a file for analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%; border: 1px solid #D8F3DC; transition: all 0.3s ease;">
        <div style="background: #D8F3DC; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; transition: transform 0.3s ease;">
        <i class="fas fa-cog" style="font-size: 1.5rem; color: #1B4332;"></i>
        </div>
        <h3 style="color: #1F2937; font-size: 1.1rem; text-align: center; margin-bottom: 0.75rem;">2. Analyze</h3>
        <p style="color: #4B5563; font-size: 0.9rem; text-align: center; line-height: 1.5; margin: 0;">Our AI analyzes the content using advanced algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%; border: 1px solid #D8F3DC; transition: all 0.3s ease;">
        <div style="background: #D8F3DC; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; transition: transform 0.3s ease;">
        <i class="fas fa-check" style="font-size: 1.5rem; color: #1B4332;"></i>
        </div>
        <h3 style="color: #1F2937; font-size: 1.1rem; text-align: center; margin-bottom: 0.75rem;">3. Get Result</h3>
        <p style="color: #4B5563; font-size: 0.9rem; text-align: center; line-height: 1.5; margin: 0;">Receive a clear report on whether the news is fake or real.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# Analysis Section
st.markdown("<a id='analyze'></a>", unsafe_allow_html=True)
st.markdown("## Detect Fake News")

# Initialize session state
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# Create the form
with st.form("news_form"):
    # Text area
    text_input = st.text_area(
        "Paste the news article here for analysis",
        height=200,
        placeholder="Type or paste your news article here to check if it's fake or real. For example: 'Scientists discover unicorns living in the Amazon rainforest.'"
    )
    
    # Submit button aligned to left
    col1, col2 = st.columns([1, 3])
    with col1:
        submitted = st.form_submit_button("Analyze Article", type="primary", use_container_width=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

# Process uploaded file
if uploaded_file is not None:
    text_input = uploaded_file.getvalue().decode("utf-8")

# Show result if form is submitted
if submitted and text_input.strip():
    with st.spinner("Analyzing article..."):
        input_vector = vectorizer.transform([text_input])
        prediction = model.predict(input_vector)[0]
        # Get confidence score using decision function
        decision_score = model.decision_function(input_vector)[0]
        # Convert to probability-like score between 0 and 100
        confidence = min(99, max(1, int(50 + decision_score * 10)))
        
        # Display result
        result_class = "fake" if prediction == "FAKE" else "real"
        icon = "❌" if result_class == "fake" else "✅"
        result_title = "Potential Fake News Detected" if result_class == "fake" else "This News Appears to be Reliable"
        
        # Prediction box
        st.markdown(f"""
        <div style="background: {'#FEE2E2' if result_class == 'fake' else '#DCFCE7'}; 
            color: {'#B91C1C' if result_class == 'fake' else '#166534'}; 
            padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;
            text-align: center; font-size: 1.5rem; font-weight: bold;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
            {'FAKE NEWS DETECTED' if result_class == 'fake' else 'RELIABLE NEWS'}
        </div>
        
        <div class="result {result_class}" style="padding: 2rem; border-radius: 8px; 
            background: {'#FEF2F2' if result_class == 'fake' else '#F0FDF4'};">
            <div style="margin-top: 1rem; text-align: center; font-size: 1.1rem;">
                <span style="background: white; color: {'#B91C1C' if result_class == 'fake' else '#166534'}; 
                    padding: 0.5rem 1.5rem; border-radius: 2rem; font-weight: 600;">
                    {confidence}% confidence
                </span>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; border-radius: 6px; 
                border-left: 4px solid {'#DC2626' if result_class == 'fake' else '#22C55E'};
                background: {'#FEF2F2' if result_class == 'fake' else '#F0FDF4'};">
                <p style="margin: 0; color: #1F2937; font-size: 0.95rem;">
                    {'This article shows characteristics commonly found in fake news. We recommend verifying the information with trusted sources.' 
                     if result_class == 'fake' else 
                     'This article appears to be reliable based on our analysis. However, always verify information from multiple sources.'}
                </p>
            </div>
            <div style="margin-top: 1.5rem; text-align: right;">
                <a href="#" style="color: #2D6A4F; text-decoration: none; display: inline-flex; align-items: center;">
                    <span style="margin-right: 6px;">ℹ️</span> Why is this {result_class.lower()}?
                    </span>
                </a>
                <button onclick="window.location.reload()" style="background: #1B4332; color: white; border: none; 
                    padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; 
                    display: inline-flex; align-items: center;">
                    <span style="margin-right: 6px;">🔄</span> Analyze Another Article
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif submitted and not text_input.strip():
    st.warning("Please enter some text or upload a file.")
