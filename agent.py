# agent.py
from dotenv import load_dotenv
from pandasai import PandasAI
from pandasai.llm import AzureOpenAI
from langchain.agents import create_pandas_dataframe_agent
#from azure.identity import DefaultAzureCredential
#from azure.ai.textanalytics import TextAnalyticsClient
import os
import streamlit as st
import pandas as pd 
from langchain.document_loaders import CSVLoader
import matplotlib.pyplot as plt


load_dotenv()    




# Setting up the api key
import environ

#env = environ.Env()
#environ.Env.read_env()

#API_KEY = env("apikey")


def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    #credential = DefaultAzureCredential()
    #llm = OpenAI(openai_api_key=API_KEY)
    llm = AzureOpenAI(
      api_token="f769445c82844edda56668cb92806c21",
      api_base="https://aoiaipsi.openai.azure.com",
      api_version="2023-07-01-preview",
      deployment_name="gpt-35-turbo-0613")


    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)
   
    # Create a Pandas DataFrame agent.
    #pandas_ai = PandasAI(llm)
    return PandasAI(llm)
    #return create_pandas_dataframe_agent(llm, df, verbose=False)
    
#agent.py

def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
         + query
    )
    # run the prompt through the agent
    response = agent.run(prompt)
    #response = text_analytics_client.analyze_sentiment(documents=["This is a great product."])
    
    # convert the response to the string
    return response.__str__()

   




