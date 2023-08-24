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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate

# Create a file uploader

st.set_page_config(page_title='Clinical Trial Docs')
st.title('🦜🔗 Clinical Trial Docs')

uploaded_file = st.file_uploader("Upload a file" , type='txt')

def generate_response(uploaded_file,query_text):
    print("uploaded_file(name)>>>>>>>>>",uploaded_file.name)
    if uploaded_file is not None:   
        # print("uploaded>>>>>>>>>>",type(uploaded_file))

        filename = uploaded_file.name
        # This is a long document we can split up.
        with open(filename) as f:
            state_of_the_union = f.read()




        # docs = [uploaded_file.read().decode()]

        # documents = [uploaded_file.read().decode()]
        # print("type>>>>>>>>>>",type(documents))
        # loader = docs.load()
        # documents = loader.load()
        # document_string = '\n'.join(docs)
        # print("type>>>>>>>>>>",type(document_string))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000,length_function = len)
        documents = text_splitter.create_documents([state_of_the_union])
        # documents = text_splitter.split_documents(document_string)
        # documents = text_splitter.split_documents(document_string)

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings)


        #define the system message template
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



        # query = input("Enter your question : ")
        result = qa({"question": query_text})

        return result["answer"]        


query_text = st.text_input('Enter your question:', placeholder = 'Upload only text files only', disabled=not uploaded_file)
#  Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted :
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

        if result is not None:
            st.info(result[0])
