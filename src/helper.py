from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# --------------------------------
# Extract Data From PDF Files
# --------------------------------

def load_pdf_file(data):

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# --------------------------------
# Keep Only Required Metadata
# --------------------------------

def filter_to_minimal_docs(
    docs: List[Document]
) -> List[Document]:

    minimal_docs: List[Document] = []

    for doc in docs:

        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src
                }
            )
        )

    return minimal_docs


# --------------------------------
# Split Documents Into Chunks
# --------------------------------

def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(
        extracted_data
    )

    return text_chunks


# --------------------------------
# Hugging Face Embeddings
# --------------------------------

def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings