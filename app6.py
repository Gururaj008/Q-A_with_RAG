import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Set up Streamlit page
st.set_page_config(layout="wide", page_title="CSV Q&A Alternative Approach")
st.title("Talk to your CSV data")

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
else:
    st.warning("Please upload a CSV file to begin.")
