from fastapi import FastAPI, UploadFile, HTTPException, Query
from sentence_transformers import SentenceTransformer
import asyncpg
import openai
import asyncio
import os

# Initialize FastAPI app
app = FastAPI()

# Load LLM Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# PostgreSQL Connection
DATABASE_URL = os.getenv("DATABASE_URL")
async def get_db():
    return await asyncpg.connect(DATABASE_URL)
@app.post("/ingest")
async def ingest_document(file: UploadFile):
    """
    Accepts document data, generates embeddings, and stores them in the database.
    """
    db = await get_db()
    try:
        content = (await file.read()).decode("utf-8")
        chunks = [content[i:i+512] for i in range(0, len(content), 512)]
        embeddings = model.encode(chunks)

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await db.execute(
                """
                INSERT INTO document_embeddings (document_name, chunk_id, content, embedding)
                VALUES ($1, $2, $3, $4)
                """,
                file.filename, f"{file.filename}_{idx}", chunk, embedding.tolist(),
            )

        await db.execute(
            """
            INSERT INTO documents (name) VALUES ($1)
            """,
            file.filename,
        )
        return {"message": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await db.close()
@app.get("/qna")
async def qna(query: str, document_ids: list[int] = Query(None)):
    """
    Accepts user questions, retrieves relevant embeddings, and generates answers.
    """
    db = await get_db()
    try:
        # Fetch embeddings for active documents or specified document IDs
        if document_ids:
            rows = await db.fetch(
                """
                SELECT content, embedding
                FROM document_embeddings
                JOIN documents ON document_embeddings.document_name = documents.name
                WHERE documents.id = ANY($1::int[])
                """,
                document_ids,
            )
        else:
            rows = await db.fetch(
                """
                SELECT content, embedding
                FROM document_embeddings
                JOIN documents ON document_embeddings.document_name = documents.name
                WHERE documents.is_active = TRUE
                """
            )

        embeddings = [row['embedding'] for row in rows]
        contents = [row['content'] for row in rows]

        # Generate embedding for the query
        query_embedding = model.encode([query])[0]

        # Perform similarity search (e.g., cosine similarity)
        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
        top_chunks = [contents[i] for i in top_indices]

        # Generate the answer using RAG
        context = "\n".join(top_chunks)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=150,
        )
        return {"answer": response["choices"][0]["text"].strip(), "relevant_chunks": top_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await db.close()

# Helper Function for Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(a ** 2 for a in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)
