import os
import pickle
from typing import List, Optional
from pathlib import Path
import re
import unicodedata
from uuid import uuid4
import tqdm
import time

import pytesseract
from pdf2image import convert_from_path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.messages import SystemMessage

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from pinecone import Pinecone, ServerlessSpec

import tools as my_tools
from state import State
import config

console = Console()
embedder = config.embedder

pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
index = pc.Index("celso-db")

def sanitize_filename(name: str) -> str:
    name = name.rsplit('.', 1)[0]
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-]', '_', name)
    return name

def create_vectorstores(unify: bool = False, embedder = embedder, ignore: list | None = []):
    '''
    Vetoriza os documentos
    '''
    base_path = os.path.join('rag_resources', 'files_db')
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    if ignore:
        files = list(set(files) - set(ignore))

    with Progress() as progress:
        task_txt = "[green]Processando arquivos..."
        if unify:
            task_txt = "[green]Carregando arquivos..."
        task = progress.add_task(task_txt, total=len(files))

        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        for file in files:
            full_path = os.path.join(base_path, file)

            if file.endswith('.pdf'):
                loader = PyPDFLoader(full_path)
                document = loader.load()
            elif file.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(full_path)
                document = loader.load()
            elif file.endswith('.txt'):
                loader = TextLoader(full_path, encoding="UTF-8")
                document = loader.load()
            else:
                print(f'Extensão não suportada: {file}')
                progress.advance(task)
                continue

            if not unify:
                chunks = splitter.split_documents(document)
                vectorstore = FAISS.from_documents(chunks, embedder)

                sanitized_filename = sanitize_filename(file)
                output_path = os.path.join('rag_resources', 'vector_db', sanitized_filename)
                vectorstore.save_local(output_path)
            else:
                documents.extend(document)

            progress.advance(task)

        if unify:
            documents = splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents, embedder)
            dst = os.path.join('rag_resources', 'main_vector_db', 'main_db')
            vectorstore.save_local(dst)
            
#create_vectorstores(unify = True, ignore = ['Celso_EP_Zapier 2.docx'])

def main_vectorstore_db(embedder = embedder):

    base_path = os.path.join('rag_resources', 'files_db')
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

    for file in files:
            full_path = os.path.join(base_path, file)

            if file.endswith('.pdf'):
                loader = PyPDFLoader(full_path)
                document = loader.load()
            elif file.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(full_path)
                document = loader.load()
            elif file.endswith('.txt'):
                loader = TextLoader(full_path, encoding="UTF-8")
                document = loader.load()
            else:
                print(f'Extensão não suportada: {file}; atualize o código.')

poppler_path = r"C:\Users\danie\Daniel\faculdade\nias\celso\poppler\poppler-24.08.0\Library\bin"

def digitalize_pdf(src: str, dst: str, poppler: str = poppler_path):
    pdf_path = src
    pages_as_images = convert_from_path(pdf_path, poppler_path=poppler)
    full_text = ""

    #ram eater
    print("Iniciando a transcrição do PDF...")
    for image in pages_as_images:
        text = pytesseract.image_to_string(image, lang='por')
        full_text += text + "\n" 

    print("Transcrição finalizada!")

    # save
    with open(dst, 'w', encoding='utf-8') as file:
        file.write(full_text)

    print(f"Transcrição finalizada com sucesso!")
    print(f"O texto foi salvo em: {dst}")

def pinecone_index(index_name: str) -> None:
    index_name = index_name

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "chunk_text"}
            }
        )

#pinecone_index('celso-db')

class ParagraphSplitter:
    def split_documents(self, docs):
        results = []
        for doc in docs:
            for paragraph in doc.page_content.split('\n\n'):
                if paragraph.strip():
                    results.append(Document(page_content=paragraph, metadata=doc.metadata))
        return results
    
# Inicializa Pinecone
def upload_single_file_to_index(file_name: str, index: str = index) -> None:    

    # load
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_name)
    elif file_name.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_name)

    documents = loader.load()

    # divide em paragrafos
    splitter = ParagraphSplitter()
    chunks = splitter.split_documents(documents)

    # Upsert no Pinecone (embedding)
    items = [
        {
        "id": str(uuid4()),
            "metadata": {
            "chunk_text": chunk.page_content
            }
        }
        for chunk in chunks
    ]

    index.upsert(items)


def upload_dir_to_index(path: str, index: str = index, exclude: list | None = None) -> None:

    file_list = list(set(os.listdir(path)) - set(exclude or []))
    
    for file_name in tqdm.tqdm(file_list):
        full_path = os.path.join(path, file_name)

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(full_path)
        elif file_name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(full_path)
        elif file_name.endswith('.txt'):
            loader = TextLoader(full_path, encoding="UTF-8")
        else:
            continue

        documents = loader.load()

        splitter = ParagraphSplitter()
        chunks = splitter.split_documents(documents)

        items = [
            {
                "id": str(uuid4()),
                "chunk_text": chunk.page_content
            }
            for chunk in chunks
        ]

        try:
            for i in range(0, len(items), 96):
                index.upsert_records(
                    namespace = 'unified-db',
                    records = items[i:i+96])
        except Exception as err:
            print(err, '\n\nSleeping for 61.0 seconds')
            time.sleep(61.0)

#upload_dir_to_index(path = r'rag_resources\files_db', exclude = ['.~lock.Celso_EP_Zapier 2.docx#', 'Celso_EP_Zapier 2.docx',])

def pinecone_retriever(query: str): 

    output = '\n\n'

    results = index.search(
        namespace='unified-db',
        query={
            'inputs': {'text': query},
            'top_k': 1
        }
    )

    for chunk in results['result']['hits'][0]['fields']['chunk_text']:
        output += chunk

    return output
