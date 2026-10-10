
from collections import defaultdict
from threading import Lock
from uuid import uuid4
import os

from flask import Flask, render_template, request, session
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# Flask App
# --------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)


# --------------------------------
# Environment Variables
# --------------------------------

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
    embedding=embeddings,
)


# --------------------------------
# Retriever
# --------------------------------

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# --------------------------------
# Groq LLM
# --------------------------------

chatModel = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)


# --------------------------------
# Conversational Memory
# --------------------------------

conversation_store = defaultdict(list)
conversation_lock = Lock()

MAX_HISTORY_TURNS = 5


def format_history(history):
    """Format recent conversation turns for the prompts."""
    if not history:
        return "No previous conversation."

    return "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in history
    )


# --------------------------------
# Follow-up Question Rewriting
# --------------------------------

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest question into a standalone search query. "
            "Use conversation history to resolve references such as 'it', "
            "'they', and 'what about its symptoms'. Preserve the user's "
            "meaning. Do not answer the question. Return only the query.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Latest question: {question}",
        ),
    ]
)


def get_search_query(question, history):
    """Make follow-up questions suitable for document retrieval."""
    if not history:
        return question

    rewrite_chain = rewrite_prompt | chatModel

    result = rewrite_chain.invoke(
        {
            "chat_history": format_history(history),
            "question": question,
        }
    )

    rewritten_question = result.content

    if isinstance(rewritten_question, str) and rewritten_question.strip():
        return rewritten_question.strip()

    return question


# --------------------------------
# RAG Prompt
# --------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
            + "\n\nUse the conversation history to understand follow-up "
            "questions, but use the retrieved medical documents as evidence "
            "for medical claims.\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "Retrieved medical context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)


# --------------------------------
# RAG Chain
# --------------------------------

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


# --------------------------------
# Flask Routes
# --------------------------------

@app.route("/")
def index():
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid4())

    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = (request.values.get("msg") or "").strip()

    if not msg:
        return "Please enter a question.", 400

    conversation_id = session.get("conversation_id")

    if not conversation_id:
        conversation_id = str(uuid4())
        session["conversation_id"] = conversation_id

    print("User:", msg)

    # Get a snapshot of this conversation's previous turns.
    with conversation_lock:
        history = list(conversation_store[conversation_id])

    try:
        # Rewrite the question only when there is prior conversation.
        search_query = get_search_query(msg, history)

        # Retrieve using the rewritten question.
        documents = retriever.invoke(search_query)

        # Generate an answer using the retrieved documents and history.
        response = question_answer_chain.invoke(
            {
                "input": msg,
                "context": documents,
                "chat_history": format_history(history),
            }
        )

    except Exception:
        app.logger.exception("Failed to process chat message")
        return (
            "Sorry, something went wrong while processing your question. "
            "Please try again.",
            500,
        )

    # Save only successfully answered turns.
    with conversation_lock:
        conversation_store[conversation_id].append(
            {
                "user": msg,
                "assistant": response,
            }
        )

        conversation_store[conversation_id] = (
            conversation_store[conversation_id][-MAX_HISTORY_TURNS:]
        )

    print("Search query:", search_query)
    print("Response:", response)

    return str(response)


# --------------------------------
# Run Application
# --------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
    )
