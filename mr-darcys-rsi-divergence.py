import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Under Maintenance",
    page_icon="🚧",
    layout="centered"
)

# Add some simple CSS to center the text and make it look professional
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 5rem;
            text-align: center;
        }
        h1 {
            margin-bottom: 20px;
        }
        .stButton button {
            margin-top: 20px;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Display the Maintenance Message
st.title("🚧 Under Construction 🚧")

st.markdown("### We are currently updating the Trading Toolbox.")
st.markdown(
    """
    The site is temporarily offline while we deploy new features and improvements.
    \n**Please check back shortly.**
    """
)

# Optional: Add a button to refresh just in case they leave the tab open
if st.button("Check for Updates"):
    st.rerun()