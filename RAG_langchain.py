import getpass
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
print(os.environ.get("LANGSMITH_API_KEY"))

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
# from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

#Load and chunk contents
pdf_files=[
    "C:\ML\Research papers\Attention_is_all_you_need.pdf",
    "C:\ML\Research papers\BERT.pdf", 
    "C:\ML\Research papers\Contrastive Language-Image Pretraining with Knowledge Graphs.pdf",
    "C:\ML\Research papers\GPT-3.pdf",
    "C:\ML\Research papers\LLaMA.pdf"
]


all_documents=[]

for file_path in pdf_files:
    loader = PyPDFLoader(file_path, mode='single',images_inner_format='markdown-img',images_parser=RapidOCRBlobParser()) 
    docs = loader.load()
    all_documents.extend(docs)

import pprint

pages=[]
# for doc in loader.lazy_load():
#     pages.append(doc)

# print(docs[0].page_content[:100])
# pprint.pp(docs[0].metadata)

assert len(docs) == 1
print(f"Total characters: {len(all_documents[0].page_content)}")

#Splitting the data into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(all_documents)
# print(f'split document into {len(all_splits)} sub-documents')

# if not os.environ.get('GOOGLE_API_KEY'):
os.environ["GOOGLE_API_KEY"]=getpass.getpass("Enter API key for Google Gemini: ")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

print(os.environ.get("GOOGLE_API_KEY"))

from langchain.chat_models import init_chat_model
llm=init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

document_ids = vector_store.add_documents(all_splits)

document_ids[:3]

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question:str
    context:List[Document]
    answer:str

def retrive(state: State):
    retrived_docs = vector_store.similarity_search(state['question'])
    return {"context":retrived_docs}

def generate(state: State):
    docs_content= "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder=StateGraph(State).add_sequence([retrive, generate])
graph_builder.add_edge(START, "retrive")
graph = graph_builder.compile()

response=graph.invoke({"question":"What is the main innovation introduced in the 'Attention is All You Need' paper?"})
print(response["answer"])

import RAG_langchain
import streamlit as st

st.set_page_config(page_title="RAG Q&A")
st.title("RAG Based question Answering")

question = st.text_input("Ask a question: ")

response=graph.invoke({"question":"What is the main innovation introduced in the 'Attention is All You Need' paper?"})

if st.button("Get Answer"):
    if question:
        with st.spinner("generating answer..."):
            answer = response["answer"]
            st.success("Answer: ")
            st.write(answer)
    else:
        st.warning("Please enter a question")