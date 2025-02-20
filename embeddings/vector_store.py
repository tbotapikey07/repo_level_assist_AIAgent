import logging
import os
import re
import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional, Union

import google.generativeai as genai
import streamlit as st

class VectorStore:
    """Advanced vector store for efficient and robust semantic search."""
    
    EMBEDDING_MODELS = {
        'small': 'models/embedding-001',
        'medium': 'models/embedding-002',
        'large': 'models/embedding-003'
    }
    
    def __init__(
        self, 
        dimension: int = 768, 
        vector_db_path: str = "vector_dbs", 
        embedding_model: str = 'medium',
        max_text_length: int = 2000
    ):
        """Initialize vector store with enhanced configuration options.
        
        Args:
            dimension (int): Embedding dimension. Defaults to 768.
            vector_db_path (str): Path to load/save vector databases.
            embedding_model (str): Size of embedding model to use.
            max_text_length (int): Maximum text length for embeddings.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('vector_store.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.dimension = dimension
        self.max_text_length = max_text_length
        self.embedding_model = self.EMBEDDING_MODELS.get(
            embedding_model, 
            self.EMBEDDING_MODELS['medium']
        )
        
        # Ensure vector database directory exists
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        
        # Load existing database if available
        if vector_db_path:
            try:
                self.load_vector_database(vector_db_path)
            except Exception as e:
                self.logger.warning(f"Could not load existing vector database: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for better embedding quality.
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s.,;:!?()[\]{}"\'`]', '', text)
        
        # Lowercase for consistency
        text = text.lower()
        
        # Truncate to max length
        return text[:self.max_text_length]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings with multiple fallback strategies.
        
        Args:
            text (str): Text to generate embedding for
        
        Returns:
            np.ndarray: Generated embedding or zero embedding
        """
        try:
            # Validate and preprocess input
            preprocessed_text = self._preprocess_text(text)
            
            if not preprocessed_text:
                self.logger.warning("Empty text after preprocessing")
                return np.zeros((1, self.dimension), dtype=np.float32)
            
            # Retrieve API key securely
            api_key = (
                os.getenv('GOOGLE_API_KEY') or 
                st.secrets.get('GOOGLE_API_KEY')
            )
            
            if not api_key:
                raise ValueError("No Google API key found")
            
            # Configure API
            genai.configure(api_key=api_key)
            
            # Generate embedding
            result = genai.embed_content(
                model=self.embedding_model,
                content=preprocessed_text,
                task_type='retrieval_document'
            )
            
            # Validate and process embedding
            if not result or 'embedding' not in result:
                raise ValueError("No embedding generated")
            
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            # Ensure correct shape
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Resize or pad embedding
            if embedding.shape[1] != self.dimension:
                self.logger.warning(
                    f"Embedding dimension mismatch: "
                    f"{embedding.shape[1]} vs {self.dimension}"
                )
                
                padded_embedding = np.zeros(
                    (1, self.dimension), 
                    dtype=np.float32
                )
                
                padded_embedding[0, :min(embedding.shape[1], self.dimension)] = (
                    embedding[0, :min(embedding.shape[1], self.dimension)]
                )
                
                embedding = padded_embedding
            
            return embedding
        
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            return np.zeros((1, self.dimension), dtype=np.float32)
    
    def add_documents(
        self, 
        documents: List[Dict], 
        batch_size: int = 50
    ) -> Dict[str, int]:
        """Enhanced document addition with batch processing.
        
        Args:
            documents (List[Dict]): Documents to add
            batch_size (int): Number of documents to process in a batch
        
        Returns:
            Dict[str, int]: Processing statistics
        """
        stats = {
            'total': len(documents),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in batch:
                try:
                    content = doc.get('content', '')
                    
                    # Skip empty or very short documents
                    if not content or len(content) < 10:
                        stats['skipped'] += 1
                        continue
                    
                    # Generate embedding
                    embedding = self._get_embedding(content)
                    
                    # Add to FAISS index
                    self.index.add(embedding)
                    
                    # Standardize document structure
                    standardized_doc = {
                        'metadata': {
                            'path': doc.get('path', 'Unknown'),
                            'type': doc.get('type', 'Unknown'),
                            'size': doc.get('size', 0)
                        },
                        'content': content,
                        'embedding': embedding.tolist()
                    }
                    
                    self.documents.append(standardized_doc)
                    stats['processed'] += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing document: {e}")
                    stats['errors'] += 1
        
        # Log processing statistics
        self.logger.info("Document Processing Summary:")
        self.logger.info(f"Total documents: {stats['total']}")
        self.logger.info(f"Processed: {stats['processed']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        self.logger.info(f"Errors: {stats['errors']}")
        
        return stats
    
    def load_vector_database(self, vector_db_path: str):
        """Load vector database from a specified directory
        
        Args:
            vector_db_path (str): Directory containing vector database files
        """
        try:
            # Find the latest .faiss file in the directory
            faiss_files = [f for f in os.listdir(vector_db_path) if f.endswith('.faiss')]
            
            if not faiss_files:
                self.logger.warning(f"No .faiss files found in {vector_db_path}")
                return
            
            # Sort files to get the most recent/largest
            latest_faiss_file = max(faiss_files, key=lambda f: os.path.getsize(os.path.join(vector_db_path, f)))
            
            # Full path to the FAISS index file
            faiss_path = os.path.join(vector_db_path, latest_faiss_file)
            docs_path = os.path.join(vector_db_path, latest_faiss_file + '.docs')
            
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load documents if corresponding .docs file exists
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            self.logger.info(f"Successfully loaded vector database from {faiss_path}")
            self.logger.info(f"Total documents loaded: {len(self.documents)}")
        
        except Exception as e:
            self.logger.error(f"Error loading vector database: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced search with comprehensive error handling and logging."""
        self.logger.info(f"Performing enhanced vector search for query: '{query[:50]}...'")
        
        # Validate index and documents
        if self.index is None:
            self.logger.error("FAISS index is None")
            return []
        
        if self.index.ntotal == 0:
            self.logger.warning("Vector store is empty. No search possible.")
            return []
        
        try:
            # Preprocess query
            preprocessed_query = self._preprocess_text(query)
            
            # Generate query embedding
            query_embedding = self._get_embedding(preprocessed_query)
            
            # Validate embedding
            if query_embedding is None or query_embedding.size == 0:
                self.logger.error("Failed to generate query embedding")
                return []
            
            # Ensure embedding matches index dimension
            if query_embedding.shape[1] != self.index.d:
                self.logger.error(f"Embedding dimension mismatch. Expected {self.index.d}, got {query_embedding.shape[1]}")
                return []
            
            # Perform vector search with error handling
            try:
                distances, indices = self.index.search(
                    np.array(query_embedding), k
                )
            except Exception as e:
                self.logger.error(f"FAISS search error: {str(e)}")
                return []
            
            # Format and filter results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):  # Ensure valid index
                    doc = self.documents[idx]
                    
                    # Additional text-based filtering
                    content = doc.get('content', '').lower()
                    query_lower = query.lower()
                    
                    # Score based on multiple criteria
                    score = 0
                    if query_lower in content:
                        score += 1
                    if any(word in content for word in query_lower.split()):
                        score += 0.5
                    
                    # Only include results with some relevance
                    if score > 0:
                        result = {
                            'metadata': doc.get('metadata', {}),
                            'content': doc.get('content', '')[:1000],  # Limit content length
                            'distance': float(distances[0][i]) if distances is not None else 0,
                            'relevance_score': score
                        }
                        results.append(result)
            
            # Sort results by relevance and distance
            results.sort(key=lambda x: (x['relevance_score'], x['distance']), reverse=True)
            
            # Limit to top k results
            results = results[:k]
            
            # Log search results
            self.logger.info(f"Search Results:")
            for i, result in enumerate(results, 1):
                self.logger.info(f"Result {i}:")
                self.logger.info(f"  Path: {result['metadata'].get('path', 'Unknown')}")
                self.logger.info(f"  Type: {result['metadata'].get('type', 'Unknown')}")
                self.logger.info(f"  Size: {result['metadata'].get('size', 0)} bytes")
                self.logger.info(f"  Distance: {result['distance']}")
                self.logger.info(f"  Relevance Score: {result['relevance_score']}")
                self.logger.info(f"  Content Preview: {result['content'][:100]}...")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Unexpected error during vector search: {str(e)}")
            return []
    
    def save_index(self, path: str):
        """Save FAISS index and documents with comprehensive logging and error handling."""
        try:
            # Validate index and documents
            if self.index is None:
                raise ValueError("FAISS index is None")
            
            if len(self.documents) == 0:
                raise ValueError("No documents to save")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, path)
            
            # Save documents alongside index
            doc_path = path + '.docs'
            with open(doc_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            self.logger.info(f"Vector store saved successfully:")
            self.logger.info(f"  Index path: {path}")
            self.logger.info(f"  Documents path: {doc_path}")
            self.logger.info(f"  Total documents: {len(self.documents)}")
            self.logger.info(f"  Index dimension: {self.index.d}")
            self.logger.info(f"  Index total: {self.index.ntotal}")
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_index(self, path: str):
        """Load FAISS index and documents with comprehensive logging and error handling."""
        try:
            # Validate path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")
            
            # Load FAISS index
            self.index = faiss.read_index(path)
            
            # Load documents
            doc_path = path + '.docs'
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"Documents file not found: {doc_path}")
            
            with open(doc_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            self.logger.info(f"Vector store loaded successfully:")
            self.logger.info(f"  Index path: {path}")
            self.logger.info(f"  Documents path: {doc_path}")
            self.logger.info(f"  Total documents: {len(self.documents)}")
            self.logger.info(f"  Index dimension: {self.index.d}")
            self.logger.info(f"  Index total: {self.index.ntotal}")
            
            # Validate index and documents
            if self.index.ntotal != len(self.documents):
                self.logger.warning(f"Mismatch between index total ({self.index.ntotal}) and documents ({len(self.documents)})")
        
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            # Reset index and documents
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            raise
    
    def delete_vector_store(self, path: str = None):
        """
        Delete vector store files with comprehensive logging and error handling.
        
        Args:
            path (str, optional): Path to the vector store files to delete. 
                                  If None, deletes the current instance's files.
        """
        try:
            # Use default path if not provided
            if path is None:
                path = os.path.join('vector_dbs', 'testv2')
            
            # Construct file paths
            faiss_path = f"{path}.faiss"
            docs_path = f"{path}.faiss.docs"
            
            # Log deletion attempt
            self.logger.info(f"Attempting to delete vector store files:")
            self.logger.info(f"  FAISS Index: {faiss_path}")
            self.logger.info(f"  Documents: {docs_path}")
            
            # Delete FAISS index file
            if os.path.exists(faiss_path):
                os.remove(faiss_path)
                self.logger.info(f"Successfully deleted FAISS index: {faiss_path}")
            else:
                self.logger.warning(f"FAISS index not found: {faiss_path}")
            
            # Delete documents file
            if os.path.exists(docs_path):
                os.remove(docs_path)
                self.logger.info(f"Successfully deleted documents file: {docs_path}")
            else:
                self.logger.warning(f"Documents file not found: {docs_path}")
            
            # Reset current instance
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            
            self.logger.info("Vector store files deleted successfully. Instance reset.")
        
        except PermissionError:
            self.logger.error(f"Permission denied when trying to delete files at {path}")
            raise
        
        except OSError as e:
            self.logger.error(f"OS error occurred while deleting vector store files: {str(e)}")
            raise
        
        except Exception as e:
            self.logger.error(f"Unexpected error during vector store deletion: {str(e)}")
            raise
