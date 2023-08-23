import os

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate




loader = TextLoader(file_path="NCT03824977.txt")
# loader = CSVLoader(file_path="NCT03824977.csv")

# loader = TextLoader(file_path="ctg-studies-thyroid-cancer.csv")
# loader = CSVLoader(file_path="ctg-studies-thyroid-cancer.csv")
# loader = CSVLoader(file_path="NCT03824977.csv")

# data = loader.load()

# ctg-studies
documents = loader.load()
print("documents>>>>>> loaded")
#chunk and embedding

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
print("documents>>>>>> splitted")


embeddings = OpenAIEmbeddings()
print("documents>>>>>> embedded")

db = Chroma.from_documents(documents, embeddings)

#define the system message template
system_template = """
you are an intelligent clinical trail researcher and excellent at finding answers from the documents.
i will ask questions from the document and you'll help me try finding the answers from this.
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
---------------
{context}
"""

qa_prompt = PromptTemplate.from_templatclinical_trial_app1e(system_template)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,model_name = "gpt-4"), db.as_retriever(), memory=memory,condense_question_prompt=qa_prompt)
# qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0 , model_name = "chatgpt-4"), vectorstore.as_retriever())




# query = "What is the brief summary?"
# query = "Give me the chat histry?"


query = input("Enter your question : ")
result = qa({"question": query})

# response =  qa({"question" : query})


print(result["answer"])
