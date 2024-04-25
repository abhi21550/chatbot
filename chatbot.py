import os
import getpass

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
import chainlit as cl

os.environ['HUGGING_FACE_API_KEY'] = 

path=input("Enter pdf:")
loader=PyPDFLoader(path)
pages=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
docs=splitter.split_documents(pages)

embeddings=HuggingFaceEmbeddings()
doc_search=Chroma.from_documents(docs, embeddings)




model = "tiiuae/falcon-7b"
llm=HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_API_KEY'],repo_id=model, model_kwargs={'temperature':0.2,'max_length':1000})

@cl.on_chat_start
def main():
  retrieval_chain=RetrievalQA.from_chain_type(llm,chain_type='stuff',retriever=doc_search.as_retriever())
  cl.user_session.set("retrieval_chain", retrieval_chain)

@cl.on_message
async def main(message: cl.Message):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    message_content = message.content  # Extract content from the message object
    res = await retrieval_chain.acall(message_content, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["result"]).send()
