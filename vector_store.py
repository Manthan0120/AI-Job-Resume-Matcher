# vector_store.py
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.initialize_vectorstore()
    
    def initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store with automatic batching"""
        docs = []
        
        for doc in documents:
            # Validate content exists and is not empty
            content = doc.get('content', '').strip()
            if not content or len(content) < 10:
                print(f"Skipping document {doc.get('id')} - insufficient content")
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            if not chunks:
                print(f"Warning: No chunks created for document {doc.get('id')}")
                continue
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            'id': doc['id'],
                            'type': doc['type'],
                            'filename': doc.get('filename', ''),
                            'title': doc.get('title', ''),
                            'company': doc.get('company', ''),
                            'chunk_id': i
                        }
                    ))
        
        if not docs:
            print("Warning: No valid documents to add")
            return
        
        # Get max batch size from ChromaDB
        try:
            # Access the max_batch_size from the underlying ChromaDB client
            max_batch = 5000  # Safe default
            if hasattr(self.vectorstore, '_client'):
                max_batch = getattr(self.vectorstore._client, 'max_batch_size', 5000)
            elif hasattr(self.vectorstore, '_collection'):
                if hasattr(self.vectorstore._collection, '_client'):
                    max_batch = getattr(self.vectorstore._collection._client, 'max_batch_size', 5000)
        except:
            max_batch = 5000  # Fallback to safe default
        
        print(f"Max batch size: {max_batch}")
        print(f"Total chunks to add: {len(docs)}")
        
        # Add documents in batches
        total_added = 0
        for i in range(0, len(docs), max_batch):
            batch = docs[i:i + max_batch]
            batch_num = (i // max_batch) + 1
            total_batches = (len(docs) // max_batch) + 1
            
            print(f"Adding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            try:
                self.vectorstore.add_documents(batch)
                total_added += len(batch)
                print(f"Successfully added batch {batch_num}")
            except Exception as e:
                print(f"Error adding batch {batch_num}: {e}")
                # Try smaller batch if it fails
                if len(batch) > 1000:
                    print(f"Retrying with smaller batches...")
                    for j in range(0, len(batch), 1000):
                        mini_batch = batch[j:j + 1000]
                        try:
                            self.vectorstore.add_documents(mini_batch)
                            total_added += len(mini_batch)
                        except Exception as e2:
                            print(f"Error adding mini-batch: {e2}")
        
        # Persist after all batches
        self.vectorstore.persist()
        print(f"Successfully added {total_added}/{len(docs)} document chunks")
    

    
    def similarity_search(self, query: str, doc_type: str = None, k: int = 5):
        """Perform similarity search"""
        if not self.vectorstore:
            return []
        
        # Create filter for document type
        filter_dict = {"type": doc_type} if doc_type else None
        
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        return results
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            # Get embeddings for both texts
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [embedding1], 
                [embedding2]
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0