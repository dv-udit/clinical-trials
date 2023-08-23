import streamlit as st
import json
import os

from langchain.llms import OpenAI
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit

import json

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Create a file uploader

st.set_page_config(page_title='Clinical Trial Docs')
st.title('🦜🔗 Clinical Trial Docs')

uploaded_file = st.file_uploader("Upload a file")



def generate_response(uploaded_file, query_text):

    if uploaded_file is not None:
        # Read and print the content of the uploaded file
        file_contents = uploaded_file.read()
        string_data = file_contents.decode('utf-8')
        file_data = json.loads(string_data)
        if len(query_text) > 4000:
            st.error("Max length exceeded")

        json_spec = JsonSpec(dict_=file_data, max_value_length=4000)
        json_toolkit = JsonToolkit(spec=json_spec)
        json_agent_executor = create_json_agent(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        toolkit=json_toolkit,
        max_execution_time=60,
        early_stopping_method="generate",
        verbose=True
        )

        # question = input("Enter your question here")
        response = json_agent_executor.run(query_text)
        # print("Result:", response)
        return response

query_text = st.text_input('Enter your question:', placeholder = '', disabled=not uploaded_file)

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



