import streamlit as st
import openai
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Variable to store conversation history
conversation_history = ""
selected_column = None  # Added variable to store the selected column

def check_api_key(openai_api_key):
    try:
        openai.api_key = openai_api_key

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=10
        )

        return response is not None

    except Exception as e:
        if 'invalid API key' in str(e):
            st.error("Incorrect API key provided. Please check your OpenAI API key.")
        else:
            st.error(f"An unexpected error occurred: {str(e)}")

        return False

def fetch_the_answer(df, openai_api_key, question):
    global conversation_history, selected_column

    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo-1106',
        max_tokens=4000,
        temperature=0.0
    )
    agent = create_pandas_dataframe_agent(chat, df, agent_executor_kwargs={'handle_parsing_errors': True})

    # Add the current question to the conversation history
    conversation_history += f"User: {question}\n"

    # Build the prompt with the entire conversation history and the selected column
    prompt1 = '''If you feel the question is incomplete or if you dont understand the question.
                please ask the user to provide you with more information.
                While answering whenever possible, put the results in the tabular form using markdown format.'''
    
    # Check if the user specified a column in the question
    column_match = re.search(r"Column: (.+?)\n", question)
    if column_match:
        selected_column = column_match.group(1)
    elif selected_column:
        # If no column specified, use the last selected column
        prompt = f"Answer the following question: {question} {prompt1}\nColumn: {selected_column}\n{conversation_history}"
    else:
        # If no column specified and no last selected column, proceed without specifying a column
        prompt = f"Answer the following question: {question} {prompt1}\n{conversation_history}"

    # Run the agent
    res = agent.run(prompt)

    # Extract the selected column from the agent's response
    column_match = re.search(r"Column: (.+?)\n", res)
    if column_match:
        selected_column = column_match.group(1)

    # Add the question, answer, and selected column to the conversation history
    conversation_history += f"Assistant: {res}\nColumn: {selected_column}\n"

    return res

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 60px;color:cyan }
                    </style>
                    <p class="custom-text"> InsightfulTalks: Transforming Data into Conversational Wisdom</p>
                    """, unsafe_allow_html=True)
    st.divider()
    st.header('About the project')
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">InsightfulTalks: "Transforming Data into Conversational Wisdom" is an innovative project that seamlessly integrates advanced language models, like OpenAI\'s GPT family, into the data exploration process. This application empowers users to engage in dynamic conversations with their datasets, fostering a natural and interactive exploration experience. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">The AI-driven conversation provides instant, context-aware responses, enabling users to extract meaningful insights efficiently. The application further enhances user experience by presenting results in markdown format, with the option to display structured information in tables for clarity. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.subheader('Steps on how to use the app')
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">1. Upload Data: Choose and upload your CSV file containing the data you want to explore. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">2. Enter API Key: Input your OpenAI API key to enable intelligent conversations with your data. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">3. Ask a Question: Pose questions about your data in the provided text box. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">4. Generate Insights: Click the button to get AI-generated answers and insights about your data. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify">5. Repeat as Needed: Continue the process to have dynamic conversations and gain deeper insights into your dataset. </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify"> </div>', unsafe_allow_html=True)
    st.write('')
    st.write('')

    # User input for CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.write('')
        st.write('')
        # User input for OpenAI API key
        with st.form(key='my_form'):
            openai_api_key = st.text_input("Enter your OpenAI API key")
            question = st.text_input("Enter the question")
            submit_button = st.form_submit_button(label='Proceed to authenticate the API key and generate answer')
            if submit_button:
                Flag = check_api_key(openai_api_key)
                if Flag == True:
                    st.success('API key verified, Proceeding to answering the question ')
                    if question is not None:
                        df = pd.read_csv(uploaded_file)
                        result = fetch_the_answer(df, openai_api_key, question)
                        st.markdown(f"**Answer:**\n\n{result}", unsafe_allow_html=True)
    col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
    with col1005:
        st.markdown("""
                                <style>
                                @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                                .custom-text-10 { font-family: 'Agdasima', sans-serif; font-size: 28px;color: cyan  }
                                </style>
                                <p class="custom-text-10">An Effort by : MAVERICK_GR </p>
                                """, unsafe_allow_html=True) 
