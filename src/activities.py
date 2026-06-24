from temporalio import activity
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams
from zeroentropy import ZeroEntropy
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid
from typing import List
import pdfplumber
import semchunk
import tiktoken

@activity.defn
async def extract(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

@activity.defn
async def chunk(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    MAX_TOKENS = 500
    chunker = semchunk.chunkerify(encoding, MAX_TOKENS)
    return chunker(text)

@activity.defn
async def index_dense(params: dict) -> str:
    chunks = params["chunks"]
    file_path = params["file_path"]
    qdrant_client = QdrantClient(url="http://localhost:6333")

    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings_list = []

    for chunk in chunks:
        response = openai_client.embeddings.create(
            input=chunk,
            model="text-embedding-3-large"
        )
        embeddings_list.append(response.data[0].embedding)

    # qdrant_client.recreate_collection(
    #     collection_name="documents_dense",
    #     vectors_config=VectorParams(
    #         size=len(embeddings_list[0]),  # embedding dimension
    #         distance="Cosine"
    #     )
    # )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings_list[i],
            payload={
                "text": chunks[i],
                "source": file_path,
                "chunk_index": i
            }
        )
        for i in range(len(chunks))
    ]

    qdrant_client.upsert(
        collection_name="documents_dense",
        points=points
    )

    return "success"

@activity.defn
async def retrieve_dense(prompt: str) -> List[float]:
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-large"
    )
    prompt_dense_vector = response.data[0].embedding
    return prompt_dense_vector

@activity.defn
async def query(params: dict) -> List[dict]:
    prompt_vector = params["prompt_vector"]
    collection_name = params["collection_name"]
    client = QdrantClient(url="http://localhost:6333")

    result = client.search(
        collection_name=collection_name,
        query_vector=prompt_vector,
        limit=10
    )

    return [r.dict() for r in result]

@activity.defn
async def rerank(params: dict) -> List[dict]:
    result = params["result"]
    prompt = params["prompt"]

    documents = [
        r["payload"]["text"]
        for r in result
        if "payload" in r and "text" in r["payload"]
    ]

    zclient = ZeroEntropy(api_key=os.getenv("OPENENTROPY_API_KEY"))

    response = zclient.models.rerank(
        model="zerank-1",
        query=prompt,
        documents=documents
    )

    print(response.model_dump_json(indent=4))

    reranked = [
        {"text": documents[item.index], "score": item.relevance_score}
        for item in response.results
    ]

    top3 = sorted(reranked, key=lambda x: x["score"], reverse=True)[:3]

    return top3

@activity.defn
async def generate(params: dict) -> str:
    reranked_result = params["reranked_result"]
    user_prompt = params["prompt"]
    
    top_texts = [item["text"] for item in reranked_result[:3]]

    context_block = "\n\n---\n\n".join(top_texts)

    final_prompt = f"""
        You are a helpful assistant. Use the following retrieved context to answer the user's question concisely and factually.

        Context:
        {context_block}

        Question:
        {user_prompt}

        Answer:
            """.strip()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert assistant specialized in accurate text synthesis."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()

    print("\n--- Generated Answer ---\n", answer, "\n------------------------")

    return answer