import streamlit as st
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import traceback # Import traceback for detailed error printing

# --- Session State Initialization ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

# --- Model and Agent Initialization (Cached) ---

# Cache the chat model
@st.experimental_singleton
def get_chat_model():
    print("Initializing Chat Model...") # Add print statement to see when it runs
    return ChatGoogleGenerativeAI(
        api_key=st.secrets["GOOGLE_API_KEY"],
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.0,
        # max_tokens=4000 # Consider adjusting if needed, but default might be fine
        # convert_system_message_to_human=True # Might be needed for some older Langchain versions/models
    )

# Cache the agent based on the DataFrame content
# st.experimental_memo hashes the arguments to decide whether to rerun.
# It should handle pandas DataFrames by hashing their content.
@st.experimental_memo
def get_agent(_df: pd.DataFrame): # Type hint helps clarity
    print("Creating Agent...") # Add print statement to see when it runs
    chat = get_chat_model()
    # --- KEY CHANGE: Enable verbose output for debugging ---
    return create_pandas_dataframe_agent(
        llm=chat,
        df=_df.copy(), # Use a copy to ensure the original df passed to memo isn't mutated by the agent
        verbose=True,  # <<< THIS IS CRITICAL FOR DEBUGGING >>>
        agent_executor_kwargs={
            'handle_parsing_errors': True # Keep this, it tries to gracefully handle some parsing errors
            # 'handle_parsing_errors': "Check your output and make sure it conforms!" # Example of custom error message
            # 'handle_parsing_errors': False # Set to False to see the raw parsing error directly (useful for debugging)
            },
        # allow_dangerous_code=True # Uncomment if agent needs to run potentially risky code (use with caution)
        )

# --- Core Logic ---

def fetch_the_answer(df: pd.DataFrame, question: str):
    """Generates an answer using the Pandas DataFrame Agent."""
    try:
        agent = get_agent(df) # Get the cached agent for this specific DataFrame

        # Append user question to history (do this *before* sending to agent)
        st.session_state.conversation_history += f"User: {question}\n"

        # --- Prompt Construction ---
        # Combine instructions, context (column), and history
        prompt_instructions = (
            "If you feel the question is incomplete or if you don't understand the question, "
            "please ask the user for clarification. While answering, whenever possible, "
            "present the results in tabular form using Markdown. Avoid hallucinating information. "
            "Base your answers strictly on the provided dataframe. "
            "Think step-by-step about how to answer the question using the dataframe."
        )

        # Include selected column context if available
        column_context = ""
        if st.session_state.selected_column:
             column_context = f"Focus your analysis primarily on the column: {st.session_state.selected_column}\n"

        # Combine everything into the final prompt
        full_prompt = (
            f"{prompt_instructions}\n"
            f"{column_context}" # Add column context if present
            f"Conversation History:\n{st.session_state.conversation_history}\n" # Pass history
            f"Current Question: {question}" # The actual question
        )

        # --- FOR DEBUGGING: Option to Simplify Prompt ---
        # Comment out the `full_prompt` line above and uncomment the line below
        # to send only the raw question, isolating prompt complexity issues.
        # full_prompt = question
        # ---

        st.write("--- DEBUG: Final Prompt Sent to Agent ---")
        st.code(full_prompt, language='text')
        st.write("--- End Debug Prompt ---")


        # --- Run the Agent ---
        # The 'list index out of range' error likely occurs inside this call
        # due to the agent failing to parse the LLM's response.
        # `verbose=True` (set in get_agent) will show the LLM's raw output in the console.
        response = agent.run(full_prompt)

        # --- Process Response ---
        if not response:
            return "The model returned an empty response. Please try again or rephrase your question."

        # Attempt to extract column if mentioned in the response (optional)
        column_match = re.search(r"Column: (.+?)\n", response)
        if column_match:
            st.session_state.selected_column = column_match.group(1).strip()

        # Update conversation history with the assistant's response
        st.session_state.conversation_history += f"Assistant: {response}\n"
        if st.session_state.selected_column:
             st.session_state.conversation_history += f"(Implicitly focusing on Column: {st.session_state.selected_column})\n"

        return response

    except Exception as e:
        st.error("An error occurred while processing your question.") # User-friendly message

        # --- Enhanced Error Logging for Debugging ---
        st.write("--- DEBUG: Exception Details ---")
        st.write(f"Error Type: {type(e)}")
        st.write(f"Error Message: {e}")
        st.write("Traceback:")
        st.code(traceback.format_exc()) # Print the full traceback
        st.write("--- End Debug Exception ---")

        # Provide specific feedback if it's the known index error
        if "list index out of range" in str(e):
            return ("The agent encountered an internal error trying to understand the response. "
                    "This might be due to the complexity of the question or temporary API issues. "
                    "Try rephrasing your question or simplifying it. "
                    "(Check the console/terminal for `verbose=True` output from the agent for more details).")
        else:
            # Generic error for other issues
            return f"An unexpected error occurred: {str(e)}"

# --- Streamlit App UI ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="InsightfulTalks")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 50px; color: cyan; }
        .custom-text-10 { font-family: 'Agdasima', sans-serif; font-size: 20px; color: cyan; }
        </style>
        <p class="custom-text">InsightfulTalks: Transforming Data into Conversational Wisdom</p>
    """, unsafe_allow_html=True)

    st.divider()

    with st.expander("About & How to Use", expanded=False):
        st.header('About the project')
        st.markdown(
            '<div style="text-align: justify">InsightfulTalks is an innovative project that integrates advanced language models into data exploration. Engage in dynamic conversations with your dataset to extract meaningful insights using a Pandas DataFrame agent.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="text-align: justify">Results are presented in Markdown format, and structured tables will be used whenever possible. The agent uses Google\'s Gemini Pro model.</div>',
            unsafe_allow_html=True
        )

        st.subheader('Steps on how to use the app')
        st.markdown('<div style="text-align: justify">1. ⬆️ Upload your CSV file containing the data.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify">2. 🤔 Enter your question about the data in the text box (e.g., "What is the average age?", "Show me the rows where city is London", "How many unique countries are there?").</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify">3. 💬 Click "Generate Answer" to get AI-generated insights. The agent will figure out the Pandas code needed.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify">4. 🔄 Ask follow-up questions! The conversation history is maintained. Use the "Reset Conversation" button to start fresh.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify">5. 🐞 **Debugging**: If errors occur, check the terminal where you ran Streamlit. The `verbose=True` setting prints the agent\'s thought process, helping diagnose issues.</div>', unsafe_allow_html=True)

    st.write('') # Spacer

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Controls")
        # Reset conversation button
        if st.button("🔄 Reset Conversation History"):
            st.session_state.conversation_history = ""
            st.session_state.selected_column = None
            st.success("Conversation history and selected column reset.")
            # Clear agent cache if desired (might force re-creation)
            # get_agent.clear() # Uncomment cautiously if needed

        st.info("Ensure required libraries are installed:\n"
                "`pip install streamlit pandas langchain langchain-experimental langchain-google-genai google-generativeai tabulate`")


    # --- Main Area ---
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    df_display_area = st.empty() # Placeholder for dataframe
    results_area = st.container() # Container for results

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            with df_display_area:
                 st.success("CSV Loaded Successfully!")
                 st.write("DataFrame Head:")
                 st.dataframe(df.head()) # Display DataFrame head

            st.header("2. Ask a Question")
            # Use a form for better control over submission
            with st.form(key='query_form'):
                default_question = "Provide a summary of the data." if df is not None else ""
                question = st.text_area("Enter your question about the data:", value=default_question, height=100)
                submit_button = st.form_submit_button(label='💬 Generate Answer')

                if submit_button and question:
                    with st.spinner("🧠 Thinking... Analyzing data and generating response..."):
                         # --- Call the main function ---
                         result = fetch_the_answer(df, question)

                         # --- Display results ---
                         with results_area:
                              st.subheader("💡 Answer")
                              st.markdown(result, unsafe_allow_html=True) # Display the agent's response

                              # Display conversation history (optional)
                              # with st.expander("Show Conversation History"):
                              #    st.text_area("History", st.session_state.conversation_history, height=200, disabled=True)

        except Exception as e:
             st.error(f"Failed to read or process CSV file: {e}")
             st.code(traceback.format_exc())


    else:
         with df_display_area:
              st.warning("Please upload a CSV file to begin.")

    # --- Footer ---
    st.divider()
    st.markdown('<p style="text-align: center; color: cyan; font-family: Agdasima, sans-serif; font-size: 20px;">An Effort by : MAVERICK_GR</p>', unsafe_allow_html=True)
