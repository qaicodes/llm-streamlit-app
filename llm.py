import time, os, json, warnings, traceback
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import prompt
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

store = LocalFileStore("./cache")
embed_model_id = "BAAI/bge-small-en-v1.5"
core_embedding_model = HuggingFaceEmbeddings(model_name=embed_model_id)
embedder = CacheBackedEmbeddings.from_bytes_store(core_embedding_model,
                                                  store,
                                                  namespace=embed_model_id)

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="mps", trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

pipe = pipeline(
        "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.1, 
            top_p=0.95, 
            tok_k=40, 
            repetition_penalty=1.1
                )

llm = HuggingFacePipeline(pipeline=pipe)

def generative_llm(path, query):
    results = []
    path = "Path with all text files"
    PROMPT_TEMPLATE = """You are a text processing agent working with lease agreement document.
    Use the following pieces of information to answer the user's question.

    Context: {context}
    Question: {question}

    Only return helpful answer below and nothing else.
    helpful answer:
    """

    input_variable = ["context", "question"]
    custom_prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=input_variable)

    document_name = "leases.pdf"
    filepath = os.path.join(path, document_name)
    loader = PyPDFLoader(document_name)   
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    esops_document = text_splitter.transform_documents(pages)

    # Create VectorStore
    try:
        vectorstore = FAISS.from_documents(esops_document, core_embedding_model)
        bm25_retriever = BM25Retriever.from_document(esops_document)
        bm25_retriever.k = 5
        faiss_retriever = vectorstore.as_retriever
        ensemble_retriever = EnsembleRetriever.from_retrievers([bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        fileDict = {}
        handler = StdOutCallbackHandler()

        qa_with_sources_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever = ensemble_retriever,
            callbacks=[handler],
            chain_type_kwargs={"prompts": custom_prompt},
            return_souce_documents=True
        )

        
        query = "retrieve five values: lessor or owner, 'lessor or licensor name', 'owner', ''current amendment made as of, 'landlord site id'. format response as following {\"lessor\": {}, \"lessororlicensornamel\": {}, \"owner\": {}, \"currentamendmentexecutiondate\": {}, \"landlordsiteid\": {}}"

        response = qa_with_sources_chain({"query" : query})
        print(f"Source Document : \n {response['source_documents']}")
        json1 = response['result'].replace("\\_", "_")
        print(json1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Check for errrors")



