from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import List, TypedDict
import logging
load_dotenv()

# ---------------------- SETUP LOGGING ---------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------------- STATE STRUCTURE ---------------------- #
class GraphState(TypedDict):
    question: str
    context_chunks: List[str]
    answer: str
    language : str
    translated_question: str


def translate(state: GraphState, client: Groq) -> GraphState:
    if state["language"] != "en":
        prompt = f"""You are a language expert, translate the user question to english, only return the translated question.
        user question: {state["question"]}
        translated question:

""".strip()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
                    messages= messages,

                    # The language model which will generate the completion.
                    model="llama-3.3-70b-versatile"
                )
        return {**state, "translated_question": response.choices[0].message.content.strip()}
    return {**state, "translated_question": state["question"]}

def retrieve_context(state: GraphState, index, chunks, embedder, top_k=2) -> GraphState:
    query_embedding = embedder.encode([state["translated_question"]], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return {**state, "context_chunks": retrieved_chunks}

def build_and_run_prompt(state: GraphState, client: Groq) -> GraphState:
    context = "\n\n".join(state["context_chunks"])
    prompt = f"""
You are a knowledgeable and friendly voice guide dedicated to providing visitors with accurate and engaging information strictly based on the context provided.

---------------------
Context:
{context}
---------------------

Based solely on the above context and without using any external knowledge or assumptions, give a clear, concise, and informative response to the following visitor's query.
Answer only in {state['language']}

Query: {state['question']}

Answer:
""".strip()

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
                messages= messages,

                # The language model which will generate the completion.
                model="llama-3.3-70b-versatile"
            )
    return {**state, "answer": response.choices[0].message.content.strip()}

# ---------------------- BUILD LANGGRAPH ---------------------- #
def setup_graph(index, chunks, embedder, client = Groq()):
    logging.info("Setting up LangGraph pipeline...")
    builder = StateGraph(GraphState)
    builder.add_node("translate", lambda state: translate(state, client))
    builder.add_node("retrieve", lambda state: retrieve_context(state, index, chunks, embedder))
    builder.add_node("generate", lambda state: build_and_run_prompt(state, client))
    builder.set_entry_point("translate")
    builder.add_edge("translate", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()

# ---------------------- PUBLIC LOADER ---------------------- #
def load_graph_resources():
    logging.info("Loading resources: index, chunks, and embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    graph = setup_graph(index, chunks, embedder)
    logging.info("Graph loaded and ready.")
    return graph
