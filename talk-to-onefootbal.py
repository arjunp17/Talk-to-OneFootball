import streamlit as st
import os
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader



url_list = ["https://onefootball.com/en/home",
       "https://onefootball.com/en/team/barcelona-5"]


def generate_response(url_list, openai_api_key, query_text):
    if url_list is not None:
        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        # Select LLM Model
        llm = ChatOpenAI(temperature = 0.0)
        # Load url_list
        loader = UnstructuredURLLoader(urls=url_list)
        docs = loader.load()
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = Chroma.from_documents(docs, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                verbose=False)
        return qa_stuff.run(query_text)





# Page title
st.set_page_config(page_title='🏈🔗 Talk to OneFootball')
st.title('🏈🔗 Talk to OneFootball')

# Query text
query_text = st.text_input('Enter your query:', placeholder = 'Please provide a short summary.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not query_text)
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(url_list, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
