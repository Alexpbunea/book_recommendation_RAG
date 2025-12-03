"""
Search module for book recommendations using hybrid search.
Prioritizes genre similarity over description similarity.
Uses all-MiniLM-v6 as a reranker for final result selection.
"""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np

CHROMA_PATH = "./data/"
EMBEDDING_MODEL = "aicoral048/setfit-fined-tuned-books"
RERANKER_MODEL = "aicoral048/setfit-fined-tuned-books"

# Weights for hybrid search (genre-prioritized)
GENRE_WEIGHT = 0.85
DESCRIPTION_WEIGHT = 0.15

# Search configuration
INITIAL_RESULTS = 5  # Get top 5 from ChromaDB
FINAL_RESULTS = 2    # Return top 2 after reranking


class SetFitEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using SetFit's sentence transformer."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from setfit import SetFitModel
        print("Loading SetFit embedding model...")
        setfit_model = SetFitModel.from_pretrained(model_name)
        self._model = setfit_model.model_body
        print("Embedding model loaded!")
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


class Reranker:
    """Reranker using all-MiniLM-v6 for final result selection."""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Reranker loaded!")
    
    def rerank(self, query: str, results: list[dict], top_k: int = FINAL_RESULTS) -> list[dict]:
        """
        Rerank results by computing cosine similarity between query and descriptions.
        
        Args:
            query: The user's original query
            results: List of book results from hybrid search
            top_k: Number of top results to return
            
        Returns:
            Top-k results sorted by reranker score
        """
        if not results:
            return results
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Encode all descriptions
        descriptions = [r.get('description', '') for r in results]
        desc_embeddings = self.model.encode(descriptions, normalize_embeddings=True)
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(desc_embeddings, query_embedding)
        
        # Add reranker scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(similarities[i])
        
        # Sort by reranker score (descending)
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:top_k]


class BookSearcher:
    """Hybrid search for books using ChromaDB with description embeddings and reranking."""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, embedding_model: str = EMBEDDING_MODEL):
        """Initialize the searcher with ChromaDB, embedding model, and reranker."""
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Create embedding function
        self.embedding_function = SetFitEmbeddingFunction(embedding_model)
        
        # Load description collection (primary for semantic search)
        self.description_collection = self.client.get_collection(
            "book_descriptions",
            embedding_function=self.embedding_function
        )
        
        # Initialize reranker
        self.reranker = Reranker()
        print("Search collection loaded!")
    
    def search(self, query: str, n_results: int = 5) -> dict:
        """Search books by query similarity."""
        results = self.description_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances", "documents"]
        )
        return results
    
    def search_by_author_keyword(self, author: str, n_results: int = 10) -> list[dict]:
        """Direct keyword search for books by author name in metadata."""
        # Get all documents and filter by author
        all_docs = self.description_collection.get(
            include=["metadatas"],
            limit=10000  # Get all
        )
        
        matches = []
        author_lower = author.lower()
        for i, metadata in enumerate(all_docs['metadatas']):
            book_authors = metadata.get('authors', '').lower()
            if author_lower in book_authors:
                matches.append({
                    'id': all_docs['ids'][i],
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'categories': metadata.get('categories', 'Unknown'),
                    'description': metadata.get('description', ''),
                    'average_rating': metadata.get('average_rating', 0),
                    'published_year': metadata.get('published_year', 0),
                    'score': 1.0,  # High score for direct match
                    'author_match': True
                })
        
        # Sort by rating and return top n
        matches.sort(key=lambda x: float(x.get('average_rating', 0) or 0), reverse=True)
        return matches[:n_results]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra spaces and fixing apostrophes."""
        import re
        # Fix spaced apostrophes: "Sorcerer ' s" -> "Sorcerer's"
        text = re.sub(r"\s*'\s*", "'", text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def search_by_title_keyword(self, title: str, n_results: int = 10) -> list[dict]:
        """Direct keyword search for books by title in metadata."""
        all_docs = self.description_collection.get(
            include=["metadatas"],
            limit=10000
        )
        
        matches = []
        # Normalize the search title
        title_normalized = self._normalize_text(title)
        print(f"[Title Search] Normalized title: '{title_normalized}'")
        
        for i, metadata in enumerate(all_docs['metadatas']):
            book_title = metadata.get('title', '')
            book_title_normalized = self._normalize_text(book_title)
            
            # Check if search title is contained in book title
            if title_normalized in book_title_normalized:
                matches.append({
                    'id': all_docs['ids'][i],
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'categories': metadata.get('categories', 'Unknown'),
                    'description': metadata.get('description', ''),
                    'average_rating': metadata.get('average_rating', 0),
                    'published_year': metadata.get('published_year', 0),
                    'score': 1.0,
                    'title_match': True
                })
        
        print(f"[Title Search] Found {len(matches)} matches")
        matches.sort(key=lambda x: float(x.get('average_rating', 0) or 0), reverse=True)
        return matches[:n_results]

    def hybrid_search(
        self, 
        query: str, 
        genres: Optional[list[str]] = None,
        authors: Optional[list[str]] = None,
        titles: Optional[list[str]] = None,
        n_results: int = FINAL_RESULTS
    ) -> list[dict]:
        """
        Hybrid search with keyword matching for authors/titles.
        
        Pipeline:
        1. If authors provided: Use direct keyword matching for author field
        2. If titles provided: Use direct keyword matching for title field  
        3. Otherwise: Use semantic search with genre/entity enrichment
        4. Rerank and return top results
        
        Args:
            query: User's search query
            genres: Predicted genres from SetFit classifier
            authors: Extracted author names from NER
            titles: Extracted book titles from NER
            n_results: Maximum number of results to return
            
        Returns:
            List of book dictionaries with title, authors, description, score
        """
        print(f"[Search] Query: {query}")
        print(f"[Search] Authors: {authors}, Titles: {titles}, Genres: {genres}")
        
        # PRIORITY 1: Direct keyword matching for authors
        if authors:
            print(f"[Search] Using KEYWORD MATCH for authors: {authors}")
            keyword_results = []
            for author in authors:
                keyword_results.extend(self.search_by_author_keyword(author, n_results=5))
            
            if keyword_results:
                # Remove duplicates by id
                seen_ids = set()
                unique_results = []
                for r in keyword_results:
                    if r['id'] not in seen_ids:
                        seen_ids.add(r['id'])
                        unique_results.append(r)
                
                print(f"[Search] Found {len(unique_results)} books by keyword author match")
                for r in unique_results[:n_results]:
                    print(f"  - {r['title']} by {r['authors']}")
                return unique_results[:n_results]
        
        # PRIORITY 2: Direct keyword matching for titles
        if titles:
            print(f"[Search] Using KEYWORD MATCH for titles: {titles}")
            keyword_results = []
            for title in titles:
                keyword_results.extend(self.search_by_title_keyword(title, n_results=5))
            
            if keyword_results:
                seen_ids = set()
                unique_results = []
                for r in keyword_results:
                    if r['id'] not in seen_ids:
                        seen_ids.add(r['id'])
                        unique_results.append(r)
                
                print(f"[Search] Found {len(unique_results)} books by keyword title match")
                return unique_results[:n_results]
        
        # FALLBACK: Semantic search with genre enrichment
        print(f"[Search] Using SEMANTIC SEARCH")
        
        # Build genre-enriched query (primary - 85% weight)
        genre_query = query
        if genres:
            genre_query = f"{query}. Genre: {', '.join(genres)}"
        
        # Build entity-enriched query (secondary - 15% weight)
        entity_query = query
        if authors:
            entity_query = f"{entity_query} by {', '.join(authors)}"
        if titles:
            entity_query = f"{entity_query} similar to {', '.join(titles)}"
        
        # Get more results for fusion (fetch 10 from each)
        fetch_n = 10
        
        # Search with both enriched queries
        genre_results = self.search(genre_query, n_results=fetch_n)
        entity_results = self.search(entity_query, n_results=fetch_n)
        
        # Build score maps (convert distance to similarity score)
        # ChromaDB returns L2 distance, so lower is better
        # We convert to similarity: 1 / (1 + distance)
        genre_scores = {}
        for i, book_id in enumerate(genre_results['ids'][0]):
            distance = genre_results['distances'][0][i]
            similarity = 1 / (1 + distance)
            genre_scores[book_id] = {
                'score': similarity,
                'metadata': genre_results['metadatas'][0][i]
            }
        
        entity_scores = {}
        for i, book_id in enumerate(entity_results['ids'][0]):
            distance = entity_results['distances'][0][i]
            similarity = 1 / (1 + distance)
            entity_scores[book_id] = {
                'score': similarity,
                'metadata': entity_results['metadatas'][0][i]
            }
        
        # Combine scores with weights (genre-prioritized hybrid)
        all_book_ids = set(genre_scores.keys()) | set(entity_scores.keys())
        
        combined_results = []
        for book_id in all_book_ids:
            genre_score = genre_scores.get(book_id, {}).get('score', 0)
            entity_score = entity_scores.get(book_id, {}).get('score', 0)
            
            # Weighted combination (genre is prioritized)
            combined_score = (GENRE_WEIGHT * genre_score) + (DESCRIPTION_WEIGHT * entity_score)
            
            # Get metadata from whichever result has it
            metadata = genre_scores.get(book_id, {}).get('metadata') or \
                      entity_scores.get(book_id, {}).get('metadata', {})
            
            # AUTHOR BOOST: If user asked for specific author, boost books by that author
            book_authors = metadata.get('authors', '').lower()
            author_match = False
            if authors:
                for requested_author in authors:
                    if requested_author.lower() in book_authors:
                        combined_score *= 2.0  # Double the score for author matches
                        author_match = True
                        break
            
            combined_results.append({
                'id': book_id,
                'title': metadata.get('title', 'Unknown'),
                'authors': metadata.get('authors', 'Unknown'),
                'categories': metadata.get('categories', 'Unknown'),
                'description': metadata.get('description', ''),
                'average_rating': metadata.get('average_rating', 0),
                'published_year': metadata.get('published_year', 0),
                'score': combined_score,
                'genre_score': genre_score,
                'entity_score': entity_score,
                'author_match': author_match
            })
        
        # Sort by combined score (descending) and get top INITIAL_RESULTS (5)
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = combined_results[:INITIAL_RESULTS]
        
        # Rerank top 5 candidates using all-MiniLM-v6 and return top 2
        reranked_results = self.reranker.rerank(query, top_candidates, top_k=n_results)
        
        print(f"[Reranker] Top {len(reranked_results)} after reranking:")
        for r in reranked_results:
            author_flag = " [AUTHOR MATCH]" if r.get('author_match') else ""
            print(f"  - {r['title']} by {r['authors']}{author_flag} (rerank: {r['rerank_score']:.4f})")
        
        return reranked_results
    
    def format_results_for_prompt(self, results: list[dict]) -> str:
        """Format search results as context for the LLM prompt."""
        if not results:
            return "No relevant books found in the database."
        
        formatted = []
        for i, book in enumerate(results, 1):
            formatted.append(
                f"Book {i}: \"{book['title']}\" by {book['authors']}\nDescription: {book['description']}"
            )
        
        return "\n\n".join(formatted)


# Singleton instance for reuse
_searcher_instance: Optional[BookSearcher] = None


def get_searcher() -> BookSearcher:
    """Get or create the singleton BookSearcher instance."""
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = BookSearcher()
    return _searcher_instance


def search_books(
    query: str,
    genres: Optional[list[str]] = None,
    authors: Optional[list[str]] = None,
    titles: Optional[list[str]] = None
) -> list[dict]:
    """
    Convenience function for searching books.
    
    Args:
        query: User's search query
        genres: Predicted genres from classifier
        authors: Extracted author names
        titles: Extracted book titles
        
    Returns:
        List of top 2 matching books after reranking
    """
    searcher = get_searcher()
    return searcher.hybrid_search(query, genres, authors, titles)


if __name__ == "__main__":
    # Test the search
    searcher = BookSearcher()
    
    # Test queries
    test_queries = [
        ("I want a fantasy book with magic", ["Fantasy"], None, None),
        ("Something like Harry Potter", ["Fantasy", "Young Adult"], ["J.K. Rowling"], ["Harry Potter"]),
        ("A mystery thriller with detective", ["Mystery"], None, None),
    ]
    
    for query, genres, authors, titles in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Genres: {genres}, Authors: {authors}, Titles: {titles}")
        print(f"{'='*60}")
        
        results = searcher.hybrid_search(query, genres, authors, titles)
        print(searcher.format_results_for_prompt(results))
