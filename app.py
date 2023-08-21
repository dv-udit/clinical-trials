import os

from langchain.document_loaders import DirectoryLoader, TextLoader
DRIVE_FOLDER = "JSON_files"
loader = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', show_progress=True, loader_cls=TextLoader)

documents = loader.load()

from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


llm = OpenAI(model_name='chatgpt-4', temperature=0)


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))
     


embedding = OpenAIEmbeddings()


db = Chroma.from_documents(docs,embedding)


retriever = db.as_retriever(search_type = "similarity",search_kwargs = {"k" : 2})
qa = RetrievalQA.from_chain_type(
    llm = OpenAI() , chain_type = "map_reduce",retriever = retriever, return_source_documents = True
)

query = "Tell me something about covid data?"
result = qa({"query" : query})


retriever.get_relevant_documents(query)

print(result)
print(result["result"])
