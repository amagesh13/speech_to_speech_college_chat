import os
import json

from dotenv import load_dotenv

load_dotenv()

from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS

history = ChatMessageHistory()

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="Test03",
    model_name="gpt-3.5-turbo")


conversation = ConversationChain(
    llm = llm, verbose = 0, memory = ConversationBufferMemory()
)

def Textbot(prompt):
    response = conversation.predict(input=prompt)
    return response


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye", "end"]:
            break

        response = Textbot(user_input)
        print("Chatbot: ", response)
