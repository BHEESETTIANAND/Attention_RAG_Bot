# importing the libraries

import streamlit as st
import numpy as np
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
import cohere
from langchain.embeddings import CohereEmbeddings
from langchain_core.messages import ChatMessage,HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import nltk
nltk.download("punkt")
import os
from dotenv import load_dotenv
import time

load_dotenv()

# loading the data
loader=PyPDFLoader("1706.03762v7.pdf")
text=loader.load_and_split()

# splitting the data
text_splitter=NLTKTextSplitter(chunk_size=200,chunk_overlap=30)
chunks=text_splitter.split_documents(text)

# loading the models
load_dotenv()
cohere_api_key = os.getenv('COHERE_API_KEY')
google_api_key=os.getenv('GOOGLE_API_KEY')

# Initialize the CohereEmbeddings object
cohere_embeddings = CohereEmbeddings(
    model="embed-multilingual-v2.0",
    cohere_api_key=cohere_api_key
)

db=Chroma.from_documents(chunks,cohere_embeddings,persist_directory="./chroma_db_")
# persist the database on drive
db.persist()

# creating the connection to the vector database
db_connection=Chroma(persist_directory="./chroma_db_",embedding_function=cohere_embeddings)
# creating the retriver from the database
retriver=db_connection.as_retriever(search_kwargs={"k":1})




# Define chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_model = ChatGoogleGenerativeAI(google_api_key=google_api_key,
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()
rag_chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)
# st.title("Attention bot")


# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         response = rag_chain.invoke(prompt)

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

st.title("AT BOT")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

