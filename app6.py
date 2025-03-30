import streamlit as st
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Initialize session state for conversation history and selected column
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

# Cache the chat model so it isn’t reinitialized every time
@st.experimental_singleton
def get_chat_model():
    return ChatGoogleGenerativeAI(
        api_key=st.secrets["GOOGLE_API_KEY"],
        model="gemini-pro",  # Changed to "gemini-pro" for stability (Suggestion 2)
        temperature=0.0,
        max_tokens=4000
    )

# Cache the agent based on the DataFrame content
@st.experimental_memo
def get_agent(df):
    chat = get_chat_model()
    return create_pandas_dataframe_agent(chat, df, agent_executor_kwargs={'handle_parsing_errors': True})

def fetch_the_answer(df, question):
    agent = get_agent(df)

    st.session_state.conversation_history = f"User: {question}\n"

    prompt_instructions = (
        "If you feel the question is incomplete or if you don't understand the question, "
        "please ask the user to provide more information. While answering, whenever possible, "
        "present the results in tabular form using Markdown."
    )
    context_text = ""

    if re.search(r"Column: (.+?)\n", question):
        match = re.search(r"Column: (.+?)\n", question)
        st.session_state.selected_column = match.group(1)
        prompt = f"Answer the following question: {question} {prompt_instructions}\n{context_text}{st.session_state.conversation_history}"
    elif st.session_state.selected_column:
        prompt = (
            f"Answer the following question: {question} {prompt_instructions}\n"
            f"Column: {st.session_state.selected_column}\n{context_text}{st.session_state.conversation_history}"
        )
    else:
        prompt = f"Answer the following question: {question} {prompt_instructions}\n{context_text}{st.session_state.conversation_history}"

    # Simplified prompt for testing (Suggestion 4 - Comment out for normal use):
    # prompt = question

    st.write("DEBUG: Prompt being sent to the agent:")
    st.code(prompt)

    try:
        res = agent.run(prompt)
        if not res:
            return "The model returned an empty response. Please try again or rephrase your question."
        column_match = re.search(r"Column: (.+?)\n", res)
        if column_match:
            st.session_state.selected_column = column_match.group(1)

        st.session_state.conversation_history += f"Assistant: {res}\nColumn: {st.session_state.selected_column}\n"
        return res

    except Exception as e:
        st.write("DEBUG: Exception details:", e)
        if "list index out of range" in str(e):
            return "The model was unable to generate a complete response. It might be due to temporary issues with the service. Please try again or rephrase your question."
        return f"An error occurred while generating an answer: {str(e)}"

# Streamlit app UI
if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 60px; color: cyan; }
        </style>
        <p class="custom-text">InsightfulTalks: Transforming Data into Conversational Wisdom</p>
    """, unsafe_allow_html=True)

    st.divider()

    st.header('About the project')
    st.markdown(
        '<div style="text-align: justify">InsightfulTalks is an innovative project that integrates advanced language models into data exploration. Engage in dynamic conversations with your dataset to extract meaningful insights.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: justify">Results are presented in Markdown format, and structured tables will be used whenever possible.</div>',
        unsafe_allow_html=True
    )

    st.subheader('Steps on how to use the app')
    st.markdown('<div style="text-align: justify">1. Upload your CSV file containing the data.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">2. Enter your question (e.g., "Which city is Alice from?").</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">3. Click "Generate Answer" to get AI-generated insights.</div>', unsafe_allow_html=True)

    st.write('')

    # Reset conversation button for testing
    if st.button("Reset Conversation"):
        st.session_state.conversation_history = ""
        st.session_state.selected_column = None
        st.success("Conversation history reset.")

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head()) # Display DataFrame head (Suggestion 3)
        with st.form(key='my_form'):
            question = st.text_input("Enter your question", value="Which city is Alice from?")
            submit_button = st.form_submit_button(label='Generate Answer')
            if submit_button and question:
                st.success('Processing your question...')
                result = fetch_the_answer(df, question)
                st.markdown(f"**Answer:**\n\n{result}", unsafe_allow_html=True)

    # Footer
    col1001, col1002, col1003, col1004, col1005 = st.columns([10, 10, 10, 10, 15])
    with col1005:
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Agdasima');
            .custom-text-10 { font-family: 'Agdasima', sans-serif; font-size: 28px; color: cyan; }
            </style>
            <p class="custom-text-10">An Effort by : MAVERICK_GR</p>
        """, unsafe_allow_html=True)
