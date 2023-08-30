from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.chains.summarize import load_summarize_chain
# from langchain.chains import LoadSummarizeChain
from langchain.chains import ChatVectorDBChain



loader = PyPDFLoader("Austin_PolicyOnlineEducation_Final.pdf")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(pages,embedding=embeddings)
llm = OpenAI(temperature=0,model_name="gpt-4")   
pdf_qa = ChatVectorDBChain.from_llm(llm,vectordb)

query = "summarise the document in 1000 words"

result = pdf_qa({"question" : query , "chat_history" : ""})
print(result["answer"])
