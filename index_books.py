"""
Script to index books into ChromaDB with embeddings for description and categories.
Uses the fine-tuned all-MiniLM-v6 model from SetFit for embeddings.
Run this once to create the persistent index.
"""

import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from datetime import datetime
from tqdm import tqdm
from utils import logger

# Use the SetFit model's base for embeddings (fine-tuned on genre classification)
EMBEDDING_MODEL = "aicoral048/setfit-fined-tuned-books"
BOOKS_CSV = "./books.csv"
CHROMA_PATH = "./data/"

# Batch size for indexing
BATCH_SIZE = 5000


class SetFitEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using SetFit's sentence transformer."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from setfit import SetFitModel
        logger.info("Loading SetFit embedding model...")
        setfit_model = SetFitModel.from_pretrained(model_name)
        self._model = setfit_model.model_body
        logger.info("Embedding model loaded!")
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


def load_books_data():
    """Load and preprocess books data"""
    logger.info(f"Loading books from {BOOKS_CSV}...")
    df = pd.read_csv(BOOKS_CSV)
    
    # Clean up data
    df = df.dropna(subset=['title', 'description'])  # Need at least title and description
    df = df.fillna({
        'authors': 'Unknown',
        'categories': 'General',
        'average_rating': 0.0,
        'num_pages': 0,
        'published_year': 0
    })
    
    logger.info(f"Loaded {len(df)} books with valid data")
    return df


def create_collections(client, embedding_function):
    """Create or get ChromaDB collections with embedding function"""
    # Collection for description embeddings
    description_collection = client.get_or_create_collection(
        name="book_descriptions",
        embedding_function=embedding_function,
        metadata={
            "description": "Book description embeddings for semantic search",
            "created": str(datetime.now())
        }
    )
    
    # Collection for genre/category embeddings
    genre_collection = client.get_or_create_collection(
        name="book_genres",
        embedding_function=embedding_function,
        metadata={
            "description": "Book genre/category embeddings for semantic search",
            "created": str(datetime.now())
        }
    )
    
    return description_collection, genre_collection


def index_books(df, description_collection, genre_collection):
    """Index all books into ChromaDB collections (embeddings auto-generated)"""
    
    # Prepare data
    ids = df['isbn13'].astype(str).tolist()
    titles = df['title'].tolist()
    authors = df['authors'].tolist()
    descriptions = df['description'].tolist()
    categories = df['categories'].tolist()
    
    # Additional metadata
    ratings = df['average_rating'].tolist()
    num_pages = df['num_pages'].tolist()
    published_years = df['published_year'].tolist()
    
    # Prepare metadata for each book
    metadatas = []
    for i in range(len(df)):
        metadatas.append({
            "title": str(titles[i]),
            "authors": str(authors[i]),
            "categories": str(categories[i]),
            "average_rating": float(ratings[i]) if pd.notna(ratings[i]) else 0.0,
            "num_pages": int(num_pages[i]) if pd.notna(num_pages[i]) else 0,
            "published_year": int(published_years[i]) if pd.notna(published_years[i]) else 0,
            "description": str(descriptions[i])[:500]  # Truncate for metadata storage
        })
    
    logger.info("Indexing into description collection (embeddings auto-generated)...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE)):
        end_idx = min(i + BATCH_SIZE, len(ids))
        description_collection.add(
            ids=ids[i:end_idx],
            documents=descriptions[i:end_idx],  # ChromaDB will generate embeddings
            metadatas=metadatas[i:end_idx]
        )
    
    logger.info("Indexing into genre collection (embeddings auto-generated)...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE)):
        end_idx = min(i + BATCH_SIZE, len(ids))
        genre_collection.add(
            ids=ids[i:end_idx],
            documents=categories[i:end_idx],  # ChromaDB will generate embeddings
            metadatas=metadatas[i:end_idx]
        )
    
    logger.info(f"Successfully indexed {len(ids)} books!")


def main():
    # Initialize ChromaDB with persistent storage
    logger.info(f"Initializing ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create embedding function
    embedding_function = SetFitEmbeddingFunction()
    
    # Load books data
    df = load_books_data()
    
    # Create collections with embedding function
    description_collection, genre_collection = create_collections(client, embedding_function)
    
    # Check if already indexed
    if description_collection.count() > 0:
        logger.warning(f"Collections already have {description_collection.count()} items.")
        response = input("Do you want to clear and re-index? (y/n): ")
        if response.lower() == 'y':
            # Delete and recreate collections
            client.delete_collection("book_descriptions")
            client.delete_collection("book_genres")
            description_collection, genre_collection = create_collections(client, embedding_function)
        else:
            logger.info("Skipping indexing. Exiting.")
            return
    
    # Index books
    index_books(df, description_collection, genre_collection)
    
    logger.info("\n✅ Indexing complete!")
    logger.info(f"Description collection: {description_collection.count()} items")
    logger.info(f"Genre collection: {genre_collection.count()} items")


if __name__ == "__main__":
    main()
