
import streamlit as st
import pandas as pd

#Page editing
st.set_page_config(
    page_title="Residential Property Prediction Tools!",
    layout="wide"  # Use the 'wide' layout
)

st.markdown(
    """
    <style>
    /* Main content background color */
    .stApp {
        background-color: #c66c6c; /* Light cardinal */
    }

    /* Sidebar background color */
    .css-1d391kg {
        background-color: #ffd700; /* Gold */
    }

    /* Optional: Adjust text color for better contrast */
    .stApp, .css-1d391kg {
        color: white; /* White text for readability */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Page Content 
st.image("AimesHomeAI.jpg", caption="This is a banner", use_container_width=True)

# Title text
st.write("# Greetings Ames, Iowa citizens! ")

st.write("##### _Welcome to our House Price Prediction tool, designed specifically for the vibrant community of Ames, Iowa. Whether you're a seasoned homeowner looking to assess your property's value or a prospective buyer exploring the market, our accurate and user-friendly tool is here to assist you. Our model leverages advanced machine learning techniques and a comprehensive dataset of historical home sales in Ames to provide reliable price estimates. By inputting key property details, you can gain valuable insights into potential market values and make informed decisions._")
st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")

col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    st.header("Did you know? ")
    st.write("Ames, Iowa, is home to a unique architectural style known as the Prairie School. This style, popularized by architects like Frank Lloyd Wright, emphasizes horizontal lines, open floor plans, and a strong connection to the natural environment. Many historic homes in Ames showcase this distinctive architectural heritage, making it a fascinating place for architecture enthusiasts to explore.")
    st.image("prairie_bobshimer.jpg", caption="Prairie style buildings", use_container_width=True)
with col2:
    st.write(" ")
with col3:
    st.header("Why our website?")
    st.write("Knowing your home's value is crucial for Ames residents, whether you're planning to sell, refinance, or simply understand your financial situation. Our model empowers you with accurate and accessible home price estimates, challenging traditional real estate practices. We offer a competitive alternative to large real estate companies, providing you with the information you need to make informed decisions about your property.")
    st.image("Real-Estate-3d-2.jpg", caption="Prairie style buildings", use_container_width=True)