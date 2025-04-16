import streamlit as st
import api  # your api.py script

st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Show title and instructions
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot using a HuggingFace AI model. "
    "You'll need a HuggingFace API token. [Get it here](https://huggingface.co/settings/tokens)."
)

# API Key input
api_key = st.text_input("Enter your HuggingFace API Key", type="password")
if not api_key:
    st.info("Please enter your HuggingFace API key to continue.", icon="🗝️")

# Question input
question = st.text_input("Your Question")

# Run only when button is clicked and inputs are provided
if st.button("Ask") and question and api_key:
    context = api.get_context()
    response = api.chat_with_gpt(question, context, api_key)

    st.write(f"**Question:** {question}")
    st.write(f"**Response:** {response.get('answer', 'No answer found')}")


# Optional exit button
if st.button("Exit"):
    st.stop()
