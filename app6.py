import streamlit as st
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI  # Remove if not needed elsewhere

# Variable to store conversation history
conversation_history = ""
selected_column = None  # Variable to store the selected column

def fetch_the_answer(df, question):
    global conversation_history, selected_column

    # Initialize the chat model using Google credentials from st.secrets
    chat = ChatGoogleGenerativeAI(
        api_key=st.secrets["GOOGLE_API_KEY"],
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.0,
        max_tokens=4000
    )
    
    # Create the dataframe agent using the chat model
    agent = create_pandas_dataframe_agent(chat, df, agent_executor_kwargs={'handle_parsing_errors': True})

    # Update conversation history with the current question
    conversation_history += f"User: {question}\n"

    # Additional instructions for the agent
    prompt_instructions = (
        "If you feel the question is incomplete or if you don't understand the question, "
        "please ask the user to provide more information. While answering, whenever possible, "
        "present the results in tabular form using Markdown."
    )
    
    # Check if the user specified a column in the question
    column_match = re.search(r"Column: (.+?)\n", question)
    if column_match:
        selected_column = column_match.group(1)
    elif selected_column:
        # If no column specified, use the last selected column in the prompt
        prompt = f"Answer the following question: {question} {prompt_instructions}\nColumn: {selected_column}\n{conversation_history}"
    else:
        # Proceed without specifying a column if none is provided
        prompt = f"Answer the following question: {question} {prompt_instructions}\n{conversation_history}"

    # Run the agent with the constructed prompt
    res = agent.run(prompt)

    # Attempt to extract a column from the agent's response (if provided)
    column_match = re.search(r"Column: (.+?)\n", res)
    if column_match:
        selected_column = column_match.group(1)

    # Update conversation history with the assistant's answer and selected column
    conversation_history += f"Assistant: {res}\nColumn: {selected_column}\n"

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
        '<div style="text-align: justify">The AI-driven conversation provides instant, context-aware responses, enabling users to extract meaningful insights efficiently. The application further enhances user experience by presenting results in Markdown format, with the option to display structured information in tables for clarity.</div>',
        unsafe_allow_html=True
    )
    
    st.subheader('Steps on how to use the app')
    st.markdown('<div style="text-align: justify">1. Upload Data: Choose and upload your CSV file containing the data you want to explore.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">2. Ask a Question: Pose questions about your data in the provided text box.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">3. Generate Insights: Click the button to get AI-generated answers and insights about your data.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify">4. Repeat as Needed: Continue the process to have dynamic conversations and gain deeper insights into your dataset.</div>', unsafe_allow_html=True)
    
    st.write('')
    
    # User input for CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.write('')
        # Form for user question input (API key input removed)
        with st.form(key='my_form'):
            question = st.text_input("Enter your question")
            submit_button = st.form_submit_button(label='Generate Answer')
            if submit_button:
                st.success('Processing your question...')
                df = pd.read_csv(uploaded_file)
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
