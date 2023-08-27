__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

from langchain.llms import OpenAI
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
    # st.session_state.uploaded_file = []

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if "QA" not in st.session_state:
    st.session_state.QA = []
    
if "key_name" not in st.session_state:
    st.session_state.key_name = ""

# LLMDATA = {}

def set_LLM(uploaded_files,key_name):
    LLMDATA = {}  
    # global LLMDATA
    if("LLMDATA" in st.session_state):
        LLMDATA = st.session_state.LLMDATA
    if uploaded_files is not None:
        combined_data = ""
        # filename = uploaded_file.name
        for file in uploaded_files:
            
        # st.session_state["current_filename"]=filename
        # st.session_state.uploaded_file.append(filename)
            if file.name not in LLMDATA:
                print("uploaded_file(name)>>>>>>>>>",file.name)
                with open(file.name) as f:
                    state_of_the_union = f.read()
                combined_data = combined_data + " " + state_of_the_union
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000,length_function = len)
        # documents = text_splitter.create_documents([state_of_the_union])
        documents = text_splitter.create_documents(combined_data)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings)
        LLMDATA[key_name] = {
            "db":db
        }
        st.session_state['LLMDATA'] = LLMDATA
        print(len(LLMDATA))

def generate_response(query_text,filename):
    LLMDATA=st.session_state.LLMDATA
    if filename in LLMDATA:
        db=LLMDATA[filename]["db"]
        system_template = """
        you are an intelligent researcher and excellent at finding answers from the documents.
        i will ask questions from the document and you'll help me try finding the answers from this.
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ---------------
        {context}
        """
        qa_prompt = PromptTemplate.from_template(system_template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,model_name = "gpt-4"), db.as_retriever(), memory=memory,condense_question_prompt=qa_prompt)
        result = qa({"question": query_text})
        # print(type(result))
        # return result["answer"]
        # print("result>>>>>>",result)  
        dict = {"question" : result["question"],"answer" : result["answer"]}
        st.session_state.QA.append(dict) 
        # return result      



def file_upload_form():
    with st.form('fileform'):
        uploaded_files = st.file_uploader("Upload a file", type='txt',accept_multiple_files=True)
        # print("uploadedFile>>>>>>>>",type(uploaded_file))
        # print("uploadedFile>>>>>>>>",uploaded_file)
        submitted = st.form_submit_button('Submit')
        if submitted:
            # st.session_state.uploaded_file = uploaded_file
            # st.session_state.uploaded_file = file
            key_name = ""
            for i in uploaded_files:
                key_name += i.name
            st.session_state.key_name = key_name
            set_LLM(uploaded_files,key_name)
            # print("uploadedFile2>>>>>>>>",type(uploaded_file))   
            # st.session_state.current_filename = uploaded_file.name

            # st.session_state.current_filename = uploaded_files.name

def query_form():
    with st.form('myform'):
        query_text = st.text_input('Enter your question:', placeholder='Enter your question here')
        submitted = st.form_submit_button('Submit', disabled=(query_text==""))
        if submitted:
            filename = st.session_state.key_name
            with st.spinner('Thinking...'):
                generate_response(query_text, filename)
                # st.write("Response:", response)
                # st.write(response)
                for i in st.session_state.QA:
                    st.write("Question : " , i["question"])
                    st.write("Answer : " , i["answer"] )
file_upload_form()
query_form()