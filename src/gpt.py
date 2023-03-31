from pathlib import Path
import os
import json
import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    Document
)
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions  import ResourceExistsError
import logging, sys
from bs4 import BeautifulSoup

load_dotenv()

if(os.getenv("RUNTIME_ENVIORNMENT") == "DEV"):
    handler = logging.StreamHandler(stream=sys.stdout)
    logger = logging.getLogger("azure")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger = logging.getLogger("azure.storage.blob")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

logger = logging.getLogger()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
#os.environ["OPENAI_API_KEY"] = "<insert api key from azure>"
openai.api_key = os.getenv("OPENAI_API_KEY")

blob_container_name = os.getenv("CONTAINER_NAME")
blob_connection_string = os.getenv("StorageConnectionString")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

llm = AzureOpenAI(deployment_name="text-davinci-003", model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
    "deployment_id": "text-davinci-003",
    "engine": "text-davinci-003",
})
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    model="text-embedding-ada-002"
    #document_model_name="text-embedding-ada-002",
    #query_model_name="text-embedding-ada-002"
))

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 48
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper
)

def clean_document(text):
    #if html
    if(bool(BeautifulSoup(text, "html.parser").find())):
        logger.log(logging.INFO, "Cleaning html request")
        soup = BeautifulSoup(text, 'lxml')
        text = soup.text
    return text

def generate_index(workspace):
    file = workspace.name+'.json'
    blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=file)

    if blob_client.exists():
        blob_data = blob_client.download_blob(logging_enable=True)
        index = GPTSimpleVectorIndex.load_from_string(blob_data.readall(), service_context=service_context)
    else:
        documents = [Document(clean_document(document)) for document in workspace.documents]
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        
        json = index.save_to_string()
        blob_client.upload_blob(json, overwrite=True)        
        
    return index

def generate_solution(workspace) -> str:
    index = generate_index(workspace)
    response = index.query(workspace.input)
    return response