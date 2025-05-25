import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
from chromadb import EmbeddingFunction
from google.generativeai import embed_content
from chromadb import PersistentClient
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Load dataset
df = pd.read_csv("amazon.csv")
df = pd.concat([df.iloc[:1279, :], df.iloc[1280:, :]])  # remove corrupted row

# Prepare docs with image and product link
def prepare_docs(df):
    return [
        f"""
        PRODUCT NAME: {row['product_name']}
        DESCRIPTION: {row['about_product']}
        PRICE: {row['discounted_price']}
        RATING: {row['rating']}
        RATING COUNT: {row['rating_count']}
        PRODUCT LINK: {row['product_link']}
        """
        for _, row in df.iterrows()
    ]

documents = prepare_docs(df)

# Embedding function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs):
        return [
            embed_content(
                model="models/text-embedding-004",
                content=doc,
                task_type="retrieval_query"
            )["embedding"]
            for doc in inputs
        ]

# Setup ChromaDB
chroma_client = PersistentClient(path="./chroma_storage")
embedding_function = GeminiEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="amazonproductdb", embedding_function=embedding_function)

# Populate ChromaDB if empty
if collection.count() == 0:
    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# Extract fields from document string
def extract_product_info(doc):
    info = {
        "name": "Unnamed",
        "description": "No description available",
        "price": "N/A",
        "rating": "N/A",
        "rating_count": "0",
        "product_link": "#"
    }

    for line in doc.strip().splitlines():
        parts = line.strip().split(":", 1)
        if len(parts) != 2:
            continue
        key, value = parts[0].strip().upper(), parts[1].strip()
        if key == "PRODUCT NAME":
            info["name"] = value
        elif key == "DESCRIPTION":
            info["description"] = value
        elif key == "PRICE":
            info["price"] = value
        elif key == "RATING":
            info["rating"] = value
        elif key == "RATING COUNT":
            info["rating_count"] = value
        elif key == "PRODUCT LINK":
            info["product_link"] = value

    return info


# Streamlit UI
st.set_page_config(page_title="Product Recommender Chatbot", page_icon="🛒")
st.title("🛒 Product Recommendation Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your product recommendation assistant. What are you looking for today?"}
    ]

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask me for product recommendations...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Finding the best products for you..."):
        results = collection.query(query_texts=[query], n_results=5)
        retrieved_docs = results['documents'][0]
        context = "\n".join(retrieved_docs)

        # Gemini generation
        prompt = f"""
You're a helpful AI assistant recommending products based on user preferences.
User query: "{query}"

Based on the following recommended products:
{context}

Respond in a helpful and friendly tone with product suggestions.
"""
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0.7})
        answer = response.text or "Sorry, I couldn't find anything for your query."

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show extracted product suggestions (basic layout, not product cards)
        st.markdown("### 🛍️ Top Product Suggestions")
        for doc in retrieved_docs:
            product = extract_product_info(doc)
            st.markdown(f"**[{product['name']}]({product['product_link']})**")
            st.markdown(f"💰 **Price:** {product['price']}")
            st.markdown(f"⭐ **Rating:** {product['rating']} ({product['rating_count']} reviews)")
            st.markdown("---")


# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your product recommendation assistant. What are you looking for today?"}
    ]
    st.rerun()
