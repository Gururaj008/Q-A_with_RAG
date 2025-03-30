import streamlit as st
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Initialize session state variables if not already present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

# Cache the chat model so it isn't reinitialized on every run
@st.experimental_singleton
def get_chat_model():
    return ChatGoogleGenerativeAI(
        api_key=st.secrets["GOOGLE_API_KEY"],
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.0,
        max_tokens=4000
    )

# Cache the creation of the agent per unique DataFrame
@st.experimental_memo
def get_agent(df):
    chat = get_chat_model()
    return create_pandas_dataframe_agent(chat, df, agent_executor_kwargs={'handle_parsing_errors': True})

def fetch_the_answer(df, question):
    # Get the agent (cached based on the DataFrame content)
    agent = get_agent(df)

    # Update conversation history with the current question
    st.session_state.conversation_history += f"User: {question}\n"

    # Additional instructions for the agent
    prompt_instructions = (
        "If you feel the question is incomplete or if you don't understand the question, "
        "please ask the user to provide more information. While answering, whenever possible, "
        "present the results in tabular form using Markdown."
    )
    
    # Build the prompt with column info (if available)
    if re.search(r"Column: (.+?)\n", question):
        # If the question includes a column, update session state
        match = re.search(r"Column: (.+?)\n", question)
        st.session_state.selected_column = match.group(1)
        prompt = f"Answer the following question: {question} {prompt_instructions}\n{st.session_state.conversation_history}"
    elif st.session_state.selected_column:
        prompt = (
            f"Answer the following question: {question} {prompt_instructions}\n"
            f"Column: {st.session_state.selected_column}\n{st.session_state.conversation_history}"
        )
    else:
        prompt = f"Answer the following question: {question} {prompt_instructions}\n{st.session_state.conversation_history}"

    try:
        # Run the agent with the constructed prompt
        res = agent.run(prompt)
    except Exception as e:
        return f"An error occurred while generating an answer: {str(e)}"

    # Optionally, extract the column from the agent's response (if provided)
    column_match = re.search(r"Column: (.+?)\n", res)
    if column_match:
        st.session_state.selected_column = column_match.group(1)

    # Update conversation history with the assistant's answer
    st.session_state.conversation_history += f"Assistant: {res}\nColumn: {st.session_state.selected_column}\n"
    return res

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
        '<div style="text-align: justify">InsightfulTalks: "Transforming Data into Conversational Wisdom" is an innovative project that seamlessly integrates advanced language models into the data exploration process. This application empowers users to engage in dynamic conversations with their datasets, fostering a natural and interactive exploration experience.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: justify">The AI-driven conversation provides instant, context-aware responses, enabling users to extract meaningful insights efficiently. Results are presented in Markdown format with structured tables when appropriate.</div>',
        unsafe_allow_html=True
    )
    
    st.subheader('Steps on how to use the app')
    st.markdown('<div style="text-align: justify">1. Upload Data: Choose and upload your CSV file containing the data you want to explore.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">2. Ask a Question: Pose questions about your data in the provided text box.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">3. Generate Insights: Click the button to get AI-generated answers and insights about your data.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">4. Repeat as Needed: Continue the process to have dynamic conversations and gain deeper insights into your dataset.</div>', unsafe_allow_html=True)
    
    st.write('')
    
    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.form(key='my_form'):
            question = st.text_input("Enter your question")
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
