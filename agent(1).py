from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd 
from pandasai import PandasAI
from langchain.document_loaders import CSVLoader
from langchain.agents import agent
import io
import json
import matplotlib.pyplot as plt

load_dotenv()

from pandasai.llm import AzureOpenAI

llm = AzureOpenAI(
  api_token="f769445c82844edda56668cb92806c21",
  api_base="https://aoiaipsi.openai.azure.com",
  api_version="2023-07-01-preview",
  deployment_name="gpt-35-turbo-0613")

pandas_ai = PandasAI(llm)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)



st.title("Chat with your CSV")
st.write("Please upload your CSV file below.")
#uploaded_file = st.file_uploader("Upload a CSV")
uploaded_file = st.sidebar.file_uploader("upload a file",type=['csv'])

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df.head(3))
  
  prompt = st.text_area("Enter your prompt:")
  
  if st.button("Generate"):
    if prompt:
       with st.spinner("Generating answer....."):
        st.write(pandas_ai.run(df,prompt=prompt))
    else:
      st.warning("please enter a prompt")


 
