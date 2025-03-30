import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="CSV Q&A Alternative Approach")
st.title("CSV Q&A: Alternative Approach")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())
        
        # Convert CSV DataFrame to text (you might want to truncate for very large files)
        csv_text = df.to_csv(index=False)
        
        # Get the user's question
        question = st.text_input("Enter your question about the CSV data:")
        
        if question:
            # Create a custom prompt template
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
            
            # Format the prompt with CSV data and the user's question
            full_prompt = prompt.format(csv_data=csv_text, question=question)
            
            st.write("--- **Debug: Full Prompt** ---")
            st.code(full_prompt, language='text')
            
            # Initialize the LLM (adjust model name as needed)
            llm = ChatGoogleGenerativeAI(
            api_key=st.secrets["GOOGLE_API_KEY"],
            model="gemini-2.5-pro-exp-03-25",  # Use a supported model name
            temperature=0.0,
            convert_system_message_to_human=True  # Add this parameter
            )
            
            # Load the QA chain (using "stuff" chain type here)
            chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
            
            # Wrap CSV text as a Document
            document = Document(page_content=csv_text)
            
            # Run the QA chain with the document and the question
            result = chain.run(input_documents=[document], question=question)
            
            st.write("### Answer:")
            st.markdown(result)
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
else:
    st.warning("Please upload a CSV file to begin.")
