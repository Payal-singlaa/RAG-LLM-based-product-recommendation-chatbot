import os
import pickle
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

df = pd.read_csv("amazon.csv")
df = pd.concat([df.iloc[:1279, :], df.iloc[1280:, :]])  # Skip corrupted row if needed

EMBEDDING_CACHE_FILE = "gemini_embeddings.pkl"
CHROMA_PERSIST_DIR = "./chroma_storage"
COLLECTION_NAME = "amazonproductdb"

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs: Documents) -> Embeddings:
        embeddings = []
        for doc in inputs:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=doc,
                task_type="retrieval_document",
                title="embedding input"
            )
            embeddings.append(response["embedding"])
        return embeddings

#document preparation
def prepare_docs(df):
    return [
        f"""
        PRODUCT NAME: {row['product_name']}
        DESCRIPTION: {row['about_product']}
        PRICE: {row['discounted_price']}
        RATING: {row['rating']}
        RATING COUNT: {row['rating_count']}
        """
        for _, row in df.iterrows()
    ]

#embedding generation or loading
if os.path.exists(EMBEDDING_CACHE_FILE):
    with open(EMBEDDING_CACHE_FILE, "rb") as f:
        doc_ids, docs, doc_embeddings = pickle.load(f)
else:
    docs = prepare_docs(df)
    embed_fn = GeminiEmbeddingFunction()
    doc_embeddings = embed_fn(docs)
    doc_ids = [str(i) for i in range(len(docs))]

    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump((doc_ids, docs, doc_embeddings), f)

#chromaDB setup
embed_fn = GeminiEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

if COLLECTION_NAME in chroma_client.list_collections():
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

if collection.count() == 0:
    batch_size = 20
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i + batch_size],
            embeddings=doc_embeddings[i:i + batch_size],
            ids=doc_ids[i:i + batch_size]
        )

#recommender function
cats = ['discounted_price', 'rating', 'rating_count', 'product_name', 'img_link', 'product_link']

def recommend_items(query: str, top_k: int = 5):
    results = collection.query(query_texts=[query], n_results=top_k)
    indices = [int(id) for id in results['ids'][0]]
    return df.loc[indices, cats].reset_index(drop=True)
