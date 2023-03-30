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

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
#os.environ["OPENAI_API_KEY"] = "<insert api key from azure>"
openai.api_key = os.getenv("OPENAI_API_KEY")

index_path = os.getenv("INDEX_PATH")

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

def generate_index(workspace):
    file = workspace.name+'.json'
    path = Path(f"{index_path}\\{file}")

    if path.is_file():
        index = GPTSimpleVectorIndex.load_from_disk(path, service_context=service_context)
    else:
        documents = [Document(document) for document in workspace.documents]
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        with open(path, 'w'):
            pass
        index.save_to_disk(path)
        
    return index

def generate_solution(workspace) -> str:
    index = generate_index(workspace)
    response = index.query(workspace.input)
    return response