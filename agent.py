#agent.py
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
#setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """
# Create an OpenAI object
llm = OpenAI(openai_api_key=API_KEY)

# Read the CSV file into a pandas dataframe
df = pd.read_csv(filename)

# Create a pandas DataFrame agent
df = pd.read_csv(filename)


