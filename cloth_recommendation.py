import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import faiss
import logging
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Optional
import csv
from io import StringIO
from dotenv import load_dotenv

load_dotenv() 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
EMBEDDING_DIMENSION = 768  # text-embedding-004 dimension
FAISS_INDEX_PATH = "dress_embeddings.faiss"
METADATA_PATH = "metadata.pkl"

class DressRecommendationSystem:
    def __init__(self, gemini_api_key: str = None, model_name: str = "models/text-embedding-004"):
        """
        Initialize the recommendation system
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: Gemini embedding model name
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        self.embedding_model = None
        self.using_gemini = False
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize FAISS index
        self.index = None
        self.metadata = {}  # Maps index position to product metadata
        
        # Load existing index if available
        self.load_index()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model (Gemini or fallback to SentenceTransformer)"""
        # Try to use Gemini first
        if self.gemini_api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                os.environ['GOOGLE_API_KEY'] = self.gemini_api_key
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    task_type="retrieval_document"
                )
                self.using_gemini = True
                logger.info(f"EmbeddingGenerator configured to use LangChain Gemini model: {self.model_name}")
            except ImportError:
                logger.error("Package 'langchain-google-genai' not found. Please install it.")
                self.using_gemini = False
            except Exception as e:
                logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
                self.using_gemini = False
        
        # Fallback to SentenceTransformer
        if not self.using_gemini:
            try:
                from sentence_transformers import SentenceTransformer
                fallback_model = "all-MiniLM-L6-v2"  # Lightweight fallback model
                self.embedding_model = SentenceTransformer(fallback_model)
                logger.info(f"Falling back to SentenceTransformer model: {fallback_model}")
            except ImportError:
                logger.error("Package 'sentence-transformers' not found. Please install it.")
                raise ImportError("Neither langchain-google-genai nor sentence-transformers is available")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
    
    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """Generate embedding for given text using Gemini or fallback model"""
        try:
            if self.using_gemini:
                if task_type == "retrieval_query":
                    embedding = self.embedding_model.embed_query(text)
                    return np.array(embedding, dtype=np.float32)
                else:
                    embeddings = self.embedding_model.embed_documents([text])
                    return np.array(embeddings[0], dtype=np.float32)
            else:
                # SentenceTransformer fallback
                embedding = self.embedding_model.encode([text], convert_to_numpy=True)
                return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def create_combined_text(self, row: pd.Series) -> str:
        """Combine description and tags for embedding generation"""
        description = str(row.get('description', ''))
        tags = row.get('tags', '[]')
        
        # Parse tags if it's a JSON string
        try:
            if isinstance(tags, str):
                tags_list = json.loads(tags)
            else:
                tags_list = tags if isinstance(tags, list) else []
        except (json.JSONDecodeError, TypeError):
            tags_list = []
        
        # Combine description and tags
        tags_text = ' '.join(tags_list) if tags_list else ''
        combined_text = f"{description} {tags_text}".strip()
        
        return combined_text
    
    def validate_csv_data(self, df: pd.DataFrame) -> bool:
        """Validate that the CSV has required columns"""
        required_columns = [
            'id', 'external_product_id', 'external_slug', 'name', 'description',
            'texture_type', 'user_id', 'vendor_id', 'status', 'tags',
            'created_date', 'updated_date', 'category_id', 'product_type_id',
            'product_group_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def process_dataset(self, csv_file_path: str = None, df: pd.DataFrame = None) -> Dict:
        """
        Process dataset and create FAISS index
        
        Args:
            csv_file_path: Path to CSV file (optional)
            df: DataFrame with product data (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load data
            if df is None:
                if csv_file_path is None:
                    raise ValueError("Either csv_file_path or df must be provided")
                df = pd.read_csv(csv_file_path)
            
            # Validate data
            if not self.validate_csv_data(df):
                raise ValueError("Invalid CSV data structure")
            
            # Filter only active products
            df_active = df[df['status'] == 'ACTIVE'].copy()
            logger.info(f"Processing {len(df_active)} active products")
            
            if len(df_active) == 0:
                raise ValueError("No active products found in dataset")
            
            # Generate embeddings
            embeddings = []
            metadata = {}
            
            for idx, row in df_active.iterrows():
                try:
                    # Create combined text for embedding
                    combined_text = self.create_combined_text(row)
                    
                    if not combined_text.strip():
                        logger.warning(f"Empty combined text for product {row['id']}")
                        continue
                    
                    # Generate embedding
                    embedding = self.generate_embedding(combined_text)
                    embeddings.append(embedding)
                    
                    # Store metadata
                    metadata[len(embeddings) - 1] = {
                        'id': int(row['id']),
                        'external_product_id': str(row['external_product_id']),
                        'name': str(row['name']),
                        'description': str(row['description']),
                        'tags': row['tags'],
                        'product_group_id': str(row['product_group_id']),
                        'texture_type': str(row['texture_type']),
                        'combined_text': combined_text
                    }
                    
                    if len(embeddings) % 100 == 0:
                        logger.info(f"Processed {len(embeddings)} products...")
                        
                except Exception as e:
                    logger.error(f"Error processing product {row.get('id', 'unknown')}: {e}")
                    continue
            
            if not embeddings:
                raise ValueError("No embeddings generated from dataset")
            
            # Create FAISS index
            embeddings_array = np.vstack(embeddings)
            
            # Use IndexFlatIP for cosine similarity (after L2 normalization)
            faiss.normalize_L2(embeddings_array)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.index.add(embeddings_array)
            self.metadata = metadata
            
            # Save index and metadata
            self.save_index()
            
            logger.info(f"Successfully created FAISS index with {self.index.ntotal} embeddings")
            
            return {
                "status": "success",
                "total_products_processed": len(df_active),
                "embeddings_created": len(embeddings),
                "index_size": self.index.ntotal
            }
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, FAISS_INDEX_PATH)
                
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.info("Index and metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                
                with open(METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                    
                logger.info(f"Loaded existing index with {self.index.ntotal} embeddings")
            else:
                logger.info("No existing index found")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
            self.metadata = {}
    
    def query_similar_products(self, query: str, top_k: int = 20) -> List[int]:
        """
        Query for similar products using natural language
        
        Args:
            query: Natural language query
            top_k: Number of similar products to retrieve initially
            
        Returns:
            List of top 5 unique product IDs
        """
        try:
            if self.index is None or not self.metadata:
                raise ValueError("No index available. Please add dataset first.")
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query, task_type="retrieval_query")
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Process results to remove duplicates by product_group_id
            seen_groups = set()
            unique_products = []
            
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # Invalid index
                    continue
                    
                product_meta = self.metadata.get(idx)
                if product_meta is None:
                    continue
                
                product_group_id = product_meta['product_group_id']
                
                # Skip if we've already seen this product group
                if product_group_id in seen_groups:
                    continue
                
                seen_groups.add(product_group_id)
                unique_products.append({
                    'id': product_meta['id'],
                    'score': float(scores[0][i]),
                    'name': product_meta['name'],
                    'product_group_id': product_group_id
                })
                
                # Stop once we have 5 unique products
                if len(unique_products) >= 5:
                    break
            
            # Extract product IDs
            product_ids = [product['id'] for product in unique_products]
            
            logger.info(f"Query: '{query}' returned {len(product_ids)} unique products")
            
            return product_ids
            
        except Exception as e:
            logger.error(f"Error querying similar products: {e}")
            raise

# Initialize the recommendation system
# Set your Gemini API key in environment variable or pass it directly
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

recommendation_system = DressRecommendationSystem(
    gemini_api_key=GEMINI_API_KEY,
    model_name="models/text-embedding-004"
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "index_loaded": recommendation_system.index is not None,
        "embeddings_count": recommendation_system.index.ntotal if recommendation_system.index else 0
    })

@app.route('/add_dataset', methods=['POST'])
def add_dataset():
    """
    Add dataset from CSV file and create embeddings
    
    Expected: CSV file with the required schema
    """
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({"error": "Please provide a valid CSV file"}), 400
        
        # Read CSV content
        csv_content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Process dataset
        result = recommendation_system.process_dataset(df=df)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in add_dataset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_recommendations():
    """
    Query for dress recommendations using natural language
    
    Expected JSON: {"query": "I want something elegant and floral for a summer brunch"}
    Returns: {"product_ids": [123, 98, 76, 45, 32]}
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({"error": "Query parameter is required"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Get recommendations
        product_ids = recommendation_system.query_similar_products(query)
        
        return jsonify({"product_ids": product_ids}), 200
        
    except Exception as e:
        logger.error(f"Error in query_recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/search_details', methods=['POST'])
def debug_search_details():
    """
    Debug endpoint to see detailed search results including scores and product info
    
    Expected JSON: {"query": "summer dress"}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query or recommendation_system.index is None:
            return jsonify({"error": "Invalid query or no index available"}), 400
        
        # Generate query embedding
        query_embedding = recommendation_system.generate_embedding(query, task_type="retrieval_query")
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = recommendation_system.index.search(query_embedding, min(20, recommendation_system.index.ntotal))
        
        # Prepare detailed results
        results = []
        seen_groups = set()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
                
            product_meta = recommendation_system.metadata.get(idx)
            if product_meta is None:
                continue
            
            is_duplicate = product_meta['product_group_id'] in seen_groups
            seen_groups.add(product_meta['product_group_id'])
            
            results.append({
                'rank': i + 1,
                'product_id': product_meta['id'],
                'name': product_meta['name'],
                'product_group_id': product_meta['product_group_id'],
                'score': float(scores[0][i]),
                'is_duplicate_group': is_duplicate,
                'description': product_meta['description'][:100] + "..." if len(product_meta['description']) > 100 else product_meta['description']
            })
        
        return jsonify({
            "query": query,
            "total_results": len(results),
            "results": results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Make sure to set your Gemini API key
    # export GEMINI_API_KEY="your-gemini-api-key"
    
    app.run(debug=True, host='0.0.0.0', port=5000)