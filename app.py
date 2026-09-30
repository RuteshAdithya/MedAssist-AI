from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from src.prompt import system_prompt

import os


# --------------------------------
# Flask App
# --------------------------------

app = Flask(__name__)


# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Check your .env file.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


# --------------------------------
# Embeddings
# --------------------------------

embeddings = download_hugging_face_embeddings()


# --------------------------------
# Pinecone
# --------------------------------

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# --------------------------------
# Retriever
# --------------------------------

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# --------------------------------
# Groq LLM
# --------------------------------

chatModel = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)


# --------------------------------
# Prompt
# --------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# --------------------------------
# RAG Chain
# --------------------------------

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# --------------------------------
# Routes
# --------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("User:", msg)

    response = rag_chain.invoke({
        "input": msg
    })

    print("Response:", response["answer"])

    return str(response["answer"])


# --------------------------------
# Run Application
# --------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )