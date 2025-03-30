import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Set up Streamlit page
st.set_page_config(layout="wide", page_title="CSV Q&A Alternative Approach")
st.title("Talk to your CSV data")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-text-2 { font-family: 'Agdasima', sans-serif; font-size: 30px; color: cyan; }
    </style>
    <p class="custom-text-2">
        Unleashing the power of AI to transform your CSV data into enlightening conversations.
    </p>
""", unsafe_allow_html=True)

st.divider()

st.header('About the project')
st.markdown(
    '<div style="text-align: justify">This project is an AI-driven application that allows you to ask questions about your CSV data and receive insightful answers. It leverages the power of Large Language Models (LLMs) to understand your questions and provide contextually relevant responses based on the content of your uploaded CSV file. This is an alternative approach to using Pandas Agents, offering a simpler way to interact with your data.</div>',
    unsafe_allow_html=True
)

st.write('')
st.subheader('How to use the app')
st.markdown("""

1.  **Upload your CSV file**: Use the file uploader below to upload the CSV file you want to analyze.
</div>
""")
st.write('')
st.markdown("""

2.  **Enter your question**: Once the CSV is uploaded and previewed, a text input box will appear. Enter your question about the data in this box. Be as specific as possible for better results.
</div>
""")
st.write('')
st.markdown("""

3.  **Get your answer**: Click the "Submit" button. The application will process your question using AI and display the answer below.
</div>
""")

st.write('')
st.divider()

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV and display preview
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())

        # Convert DataFrame to CSV text
        csv_text = df.to_csv(index=False)

        # Create a form for the question input
        with st.form("question_form"):
            question = st.text_input("Enter your question about the CSV data:")
            submit_button = st.form_submit_button("Submit")

        # If form is submitted and question provided, process it
        if submit_button and question:
            prompt_template = """
You are an expert data analyst. Answer the following question based solely on the CSV data provided.
If the information is insufficient, ask for clarification.

CSV Data:
{csv_data}

Question: {question}
Answer:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["csv_data", "question"]
            )

            # Format the prompt with CSV data and the question
            full_prompt = prompt.format(csv_data=csv_text, question=question)

            # Initialize the LLM with convert_system_message_to_human=True to avoid system message issues
            llm = ChatGoogleGenerativeAI(
                api_key=st.secrets["GOOGLE_API_KEY"],
                model="gemini-2.5-pro-exp-03-25",  # Use a supported model name
                temperature=0.0,
                convert_system_message_to_human=True
            )

            # Load the QA chain using the "stuff" chain type
            chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

            # Wrap the CSV text as a Document
            document = Document(page_content=csv_text)

            # Run the QA chain with the document and the question
            result = chain.run(input_documents=[document], question=question)

            # Display the result below
            st.write("### Answer:")
            st.markdown(result)

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
    else: # Corrected indentation: This else block is now aligned with the try block
        st.divider()
        col1001, col1002, col1003, col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                .custom-text-10 { font-family: 'Agdasima', sans-serif; font-size: 28px; color: Gold; }
                </style>
                <p class="custom-text-10">An Effort by : MAVERICK_GR</p>
            """, unsafe_allow_html=True)
else: # This else block was already correctly indented
    st.warning("Please upload a CSV file to begin.")
