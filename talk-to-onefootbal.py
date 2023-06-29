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
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import SeleniumURLLoader


url_list = ["https://onefootball.com/en/home",
       "https://onefootball.com/en/team/barcelona-5",]


def generate_response(openai_api_key, query_text):
	# Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        # Select LLM Model
        llm = ChatOpenAI(temperature = 0.0)
        # Load url_list
        loader = SeleniumURLLoader(urls=url_list)
        # Create Vector Database
        index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
        # Generate response based on the input query
        return index.query(query_text, llm=llm)





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
            response = generate_response(openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
