-- Table for storing document embeddings
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768), -- Use PGVector for storing embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for document metadata
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
