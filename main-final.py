__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate

# Create a file uploader

st.set_page_config(page_title='Clinical Trial Docs')
st.title('🦜🔗 Clinical Trial Docs')

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if "QA" not in st.session_state:
    st.session_state.QA = []
    

LLMDATA = {}

def set_LLM(uploaded_file):
    global LLMDATA
    if uploaded_file is not None:
        filename = uploaded_file.name
        st.session_state["current_filename"]=filename
        if filename not in LLMDATA:
            print("uploaded_file(name)>>>>>>>>>",filename)
            with open(filename) as f:
                state_of_the_union = f.read()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000,length_function = len)
            documents = text_splitter.create_documents([state_of_the_union])
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(documents, embeddings)
            LLMDATA[filename] = {
                "db":db
            }
            st.session_state['LLMDATA'] = LLMDATA
            print(len(LLMDATA))



def generate_response(query_text,filename):
    LLMDATA=st.session_state.LLMDATA
    if filename in LLMDATA:
        db=LLMDATA[filename]["db"]
        system_template = """
        you are an intelligent clinical trail researcher and excellent at finding answers from the documents.
        i will ask questions from the document and you'll help me try finding the answers from this.
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ---------------
        {context}
        """
        qa_prompt = PromptTemplate.from_template(system_template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,model_name = "gpt-4"), db.as_retriever(), memory=memory,condense_question_prompt=qa_prompt)
        result = qa({"question": query_text})
        # return result["answer"]   
        dict = {"question" : result["question"] , "answer" : result["answer"]}
        st.session_state.QA.append(dict)     



def file_upload_form():
    with st.form('fileform'):
        uploaded_file = st.file_uploader("Upload a file", type='txt')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.session_state.uploaded_file = uploaded_file
            set_LLM(uploaded_file)
            st.session_state.current_filename = uploaded_file.name


def query_form():
    with st.form('myform'):
        query_text = st.text_input('Enter your question:', placeholder='Enter your question here')
        submitted = st.form_submit_button('Submit', disabled=(query_text==""))
        if submitted:
            filename = st.session_state.current_filename
            with st.spinner('Thinking...'):
                generate_response(query_text, filename)
                for i in st.session_state.QA:
                    st.write("Question : " + i["question"])
                    st.write("Answer : " + i["answer"])
                
file_upload_form()
query_form()