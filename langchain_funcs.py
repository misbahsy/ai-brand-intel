import urllib.request
import os
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain, VectorDBQA
from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain, load_qa_chain
from qdrant_client import QdrantClient
from langchain.docstore.document import Document
from qdrant_func import *

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')



def download_file(url, user_id):
    # Path to the local mounted folder on the Azure VM
    folder_path = f'/home/azureuser/mydrive/{user_id}/'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Filename for the downloaded file
    filename = url.split('/')[-1]

    # Full path to the downloaded file
    file_path = os.path.join(folder_path, filename)

    # Download the file and save it to the local folder
    urllib.request.urlretrieve(url, file_path)

    print(f'Successfully downloaded file from {url} to {file_path}')
    
    return file_path



#files could be of type docx, pdf, url, txt, ppt

def load_docs(filetype, userid, **kwargs):
    if filetype == 'url':
        #need to provide a list of urls to the next step
        url_list = [kwargs['input_url']]
        loader = UnstructuredURLLoader(urls=url_list)
    else:
        save_path = download_file(kwargs['s3_path'], userid)
        loader = UnstructuredFileLoader(save_path, mode="elements")
    docs = loader.load()
    return docs, loader

def generate_embeddings(scope, department, userid, filetype, **kwargs):
    docs, loader = load_docs(filetype, userid, **kwargs)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    texts = text_splitter.split_documents(docs)

    for i, text in enumerate(texts):
        text.metadata['scope'] = scope
        text.metadata['department'] = department
    
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)

    qdrant = Qdrant.from_documents(texts, embeddings, host='localhost', collection_name=userid, prefer_grpc=True)

    return qdrant.collection_name

# def search_with_sources(qdrant, query, sources):
#     embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
#     qa_chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=openai_api_key), chain_type="stuff")
#     chain = VectorDBQAWithSourcesChain(combine_documents_chain=qa_chain, vectorstore=qdrant, return_source_documents=sources)
#     answer_w_sources = chain({"question": query}, return_only_outputs=True)
#     return answer_w_sources


def qdrant_search(query, collection_name, filter_dict,k,with_score):

    client = QdrantClient("localhost", prefer_grpc=True)
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    if with_score:
        docs = test_similarity_search_with_score(query=query, k=k, filter=filter_dict, embedding_func=embeddings.embed_query, collection_name=collection_name,client=client)
        return docs
    docs = test_similarity_search(query=query, k=3, filter=filter_dict, embedding_func=embeddings.embed_query, collection_name=collection_name,client=client)
    return docs