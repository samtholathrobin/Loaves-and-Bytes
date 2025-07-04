import os
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()
URI = os.getenv("MONGODB_URI")
client = MongoClient(URI)
res_db = client["Res_Data"]
menu_collection = res_db["Res_menus"]
embedding_model = OllamaEmbeddings(model = 'nomic-embed-text')

def semantic_filter(query: str, vector_store: FAISS, embedding_model: OllamaEmbeddings, threshold: np.float32 = 0.1, k: int = 10) -> List[Document]:
    """
    Perform a semantic search and filter by similarity threshold.

    Returns a list of tuples: (menu item name, similarity score, page content).
    """
    query_vector = embedding_model.embed_query(query)
    D, I = vector_store.index.search(np.array([query_vector]), k=k)
    similarities = 1 - D[0]  # Convert distance to similarity
    # Filtering results above threshold
    results = []
    for i, sim in zip(I[0], similarities):
        if sim >= threshold:
            doc_id = vector_store.index_to_docstore_id[i]
            doc = vector_store.docstore._dict[doc_id]
            results.append(doc)
    
    return results

def search_menu(user_query, category_filter=None, tags_filter=None):

    if user_query:
        results = semantic_filter(
            query=user_query,
            vector_store=menu_vector_store,
            embedding_model=embedding_model,
            threshold=0.2,
            k=15
        )

    else:
        results = list(menu_vector_store.docstore._dict.values())     # If no query, return all documents
    if category_filter:
        results = [doc for doc in results if doc.metadata["category"].lower() == category_filter.lower()]
    if tags_filter:
        results = [doc for doc in results if tags_filter.lower() in [tag.lower() for tag in doc.metadata["tags"].split(",")]]

    return results
