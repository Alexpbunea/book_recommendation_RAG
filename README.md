# Book Recommendation RAG

A project for generating synthetic training data to fine-tune a Retrieval-Augmented Generation (RAG) model for book recommendations. Uses OpenAI's API to create diverse query-response pairs based on a dataset of books.

## Features

- **Synthetic Data Generation**: Creates realistic user queries and corresponding recommendations using OpenAI's structured outputs.
- **Async Processing**: Generates examples concurrently with a semaphore to limit API requests (max 20 concurrent).
- **Pydantic Validation**: Ensures output conforms to a predefined schema for consistency.
- **CSV Output**: Produces a dataset ready for model fine-tuning, with columns for query, retrieved indices, and response.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book_recommendation_RAG.git
   cd book_recommendation_RAG