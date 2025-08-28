import asyncio, sys

# On Windows, prefer the Selector policy for 3rd-party libs
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure the Streamlit worker thread has an event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import required libraries
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime

# Get text from PDF
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Get chunks from text
def get_text_chunks(text,model_name):
    if model_name=="Google AI":
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# Embedding these chunks and store in vector store
def get_vector_store(text_chunks,model_name,api_key=None):
    if model_name=="Google AI":
        embeddings=GoogleGenerativeAIEmbeddings(model="text-embedding-004",google_api_key=api_key)

    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Create conversational chain using LangChain
def get_conversational_chain(model_name,vector_store=None,api_key=None):
    if model_name=="Google AI":
        prompt_template="""
          Answer the question as detailed as possible from the provided context.
          Make sure to provide all the necessary details with proper structure.
          If the answer is not in the provided context then just say that 
          "Answer is not available in the context", do not provide wrong answer \n\n
          Context:\n {context}?\n
          Question:\n {question}?\n

          Answer:
        """
        model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,google_api_key=api_key)
        prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
        chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
        return chain

# User Input
def user_input(user_question,model_name,api_key,pdf_docs,conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload relevant PDF document and provide API key")
        return
         
    text_chunks=get_text_chunks(get_pdf_text(pdf_docs),model_name)
    vector_store=get_vector_store(text_chunks,model_name,api_key)

    user_question_output=""
    response_output=""

    if model_name=="Google AI":
        embeddings=GoogleGenerativeAIEmbeddings(model="text-embedding-004",google_api_key=api_key)
        new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

        docs=new_db.similarity_search(user_question)
        chain=get_conversational_chain("Google AI",vector_store=new_db,api_key=api_key)
        response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)
        response_output=response['output_text']
        pdf_names=[pdf.name for pdf in pdf_docs] if pdf_docs else []

        conversation_history.append((user_question_output,response_output,model_name,datetime.now().strftime('%Y-%m-%d %H:%M:%S')," , ".join(pdf_names)))

    # Chat design
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )

    if len(conversation_history)==1:
        conversation_history=[]

    elif len(conversation_history) > 1:
        last_item=conversation_history[-1]
        conversation_history.remove(last_item)

    for question,answer,model_name,timestamp,pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Download conversation history
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])

        csv = df.to_csv(index=False)
        
        # Convert to base64
        b64 = base64.b64encode(csv.encode()).decode()

        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    
    # For Animation function
    st.snow()

# Entry Point function
def main():
    st.set_page_config(page_title="Ch PDFs",page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/snsupratim/"
    kaggle_profile_link = "https://www.kaggle.com/snsupratim/"
    github_profile_link = "https://github.com/snsupratim/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    model_name = st.sidebar.radio("Select the Model:", ( "Google AI"))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai/google/dev/) to get an API key.")

        if not api_key:
            st.sidebar.warning("please enter your Google API key to proceed")
            return

    with st.sidebar:
        st.title("Menu:")

        col1,col2 = st.columns(2)

        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            # Clear conversation history
            st.session_state.conversation_history = []

            # Clear user question input   
            st.session_state.user_question = None  

            # Reset Google API key
            api_key = None

            # Reset PDF document
            pdf_docs = None

        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded")
                    st.session_state.user_question = ""
                
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()
                else:
                    st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")  
    
    # User input question          
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        
        # Clear user question input 
        st.session_state.user_question = "" 

if __name__ == "__main__":
    main()