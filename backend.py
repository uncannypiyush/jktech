from fastapi import FastAPI, UploadFile, HTTPException
from sentence_transformers import SentenceTransformer
import pinecone
import uuid

# Initialize FastAPI app
app = FastAPI()

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace with a model of your choice

# Initialize Pinecone
pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')
index_name = "document-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # Dimension based on the embedding model
index = pinecone.Index(index_name)

@app.post("/ingest")
async def ingest_document(file: UploadFile):
    """
    Ingest a document, generate embeddings, and store them in the vector database.
    """
    try:
        # Read and decode the file
        content = (await file.read()).decode("utf-8")
        
        # Split content into smaller chunks for embedding
        chunk_size = 512  # Customize chunk size based on use case
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Generate embeddings for each chunk
        embeddings = model.encode(chunks)
        
        # Generate unique IDs for each chunk and upsert to Pinecone
        ids = [str(uuid.uuid4()) for _ in chunks]
        vectors = [{"id": id, "values": embedding, "metadata": {"chunk": chunk}} 
                   for id, embedding, chunk in zip(ids, embeddings, chunks)]
        index.upsert(vectors)
        
        return {"status": "success", "message": "Document ingested successfully", "chunks_processed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_documents(query: str, top_k: int = 5):
    """
    Search the vector database using query embeddings and return relevant chunks.
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query])[0]
        
        # Search the vector database
        search_results = index.query(query_embedding, top_k=top_k, include_metadata=True)
        
        return {"status": "success", "matches": search_results["matches"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
