# Book Recommendation RAG

A project for generating synthetic training data to fine-tune a Retrieval-Augmented Generation (RAG) model for book recommendations. Uses OpenAI's API to create diverse query-response pairs based on a dataset of books.

## Features

- **Synthetic Data Generation**: Creates realistic user queries and corresponding recommendations using OpenAI's structured outputs.
- **Text Classification Generation**: Generates datasets for classifying user queries into genres.
- **Async Processing**: Generates examples concurrently with a semaphore to limit API requests (max 20 concurrent).
- **Pydantic Validation**: Ensures output conforms to a predefined schema for consistency.
- **CSV Output**: Produces a dataset ready for model fine-tuning.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book_recommendation_RAG.git
   cd book_recommendation_RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generate RAG Dataset (Default)

Generates a dataset with user queries, retrieved book indices, and recommendation responses.

```bash
python main.py --mode rag
```

Output: `data/generated_dataset.csv`

### Generate Text Classification Dataset

Generates a dataset with user queries and a list of genres found in the query.

```bash
python main.py --mode classification
```

Output: `data/text_clasification.csv`

### Options

- `--mode`: Choose generation mode. Options: `rag` (default), `classification`.
- `--num_examples`: Number of examples to generate. Default: 300.

Example:
```bash
python main.py --mode classification --num_examples 100
```