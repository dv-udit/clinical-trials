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

data = loader.load()

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

# #define the system message template
# system_template = """
# you are an intelligent clinical trail researcher and excellent at finding answers from the documents.
# i will ask questions from the document and you'll help me try finding the answers from this.
# If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
# ---------------
# {context}
# """

# #create the chat prompt template

# message = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template("{question}")
# ]

# qa_prompt = ChatPromptTemplate.from_messages(messages)






memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever(), memory=memory)
# qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0 , model_name = "chatgpt-4"), vectorstore.as_retriever())




# query = "What did the president say about Ketanji Brown Jackson"
# query = "What is the brief summary?"
# query = "Give me the chat histry?"


query = input("Enter your question")
result = qa({"question": query})



# response =  qa({"question" : query})


print(result["answer"])































# from langchain import OpenAI, VectorDBQA
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma


# llm = OpenAI(model_name='chatgpt-4', temperature=0)


# def split_docs(documents,chunk_size=6000,chunk_overlap=1000):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
#   return docs

# docs = split_docs(documents)
# print(len(docs))
     


# embedding = OpenAIEmbeddings()


# db = Chroma.from_documents(docs,embedding)


# retriever = db.as_retriever(search_type = "similarity",search_kwargs = {"k" : 2})
# qa = RetrievalQA.from_chain_type(
#     llm = OpenAI() , chain_type = "map_reduce",retriever = retriever, return_source_documents = True
# )

# query = "Tell me something about covid data?"
# result = qa({"query" : query})


# retriever.get_relevant_documents(query)

# print(result)
# print(result["result"])
