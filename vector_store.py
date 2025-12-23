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
        """Add documents to vector store"""
        docs = []
        for doc in documents:
            # Split text into chunks
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
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
        
        if docs:
            self.vectorstore.add_documents(docs)
            self.vectorstore.persist()
    
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