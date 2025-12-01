import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import APIError, RateLimitError, AuthenticationError
from openai.types.responses import Response
from openai.lib._parsing._responses import type_to_text_format_param
from .setup import Provider
from utils import logger
from .ouput_schema import OutputSchema, TextClassificationSchema, NERList


class Generation:
    def __init__(self, client: Provider):
        self.async_client = client.async_client
        self.general_model = client.general_model
        
        

    async def generate_content(self, user_prompt : str, reasoning : str = "low", format = None, temperature = 0.1) -> str:
        request_id = str(uuid.uuid4())
        logger.info(f"Calling Azure OpenAI API for content generation - Request ID: {request_id}")
        
        try:
            
            text_format = type_to_text_format_param(format) if format else None

            response = await self.async_client.responses.create(
                model=self.general_model,
                reasoning={"effort": reasoning},
                input=user_prompt,
                text={"format": text_format}
            )
            
            if not hasattr(response, 'output_text') or not response.output_text:
                error_msg = f"Empty response from Azure OpenAI for request {request_id}"
                logger.error(error_msg)
                raise error_msg
            
            logger.info(f"Content generation completed successfully for request: {request_id}")
            return response

        except Exception as e:
            error_msg = f"Unexpected error for request {request_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise error_msg from e
        
    async def generate_synthetic_example(self, books_sample):
        books_info = ""
        for idx, row in books_sample.iterrows():
            
            books_info += f"""
                            Book {idx}:
                            - Title: {row['title']}
                            - Author: {row['author']}
                            - Genres: {row['genres']}
                            - Rating: {row.get('rating', 'N/A')}/5
                            {row['pages']}- Description: {row['description'][:200]}...
                            """
                    
            prompt = f"""You are generating training data for a book recommendation RAG system.

                        Given these books from a database:
                        {books_info}

                        Generate a realistic example with:
                        1. A user query (natural language, varied style):
                        - Specific: "psychological thriller set in Tokyo", "short fantasy book for commute"
                        - Vague: "something exciting", "book to cry"
                        - Constraint-based: "highly rated mystery", "long historical fiction"
                        
                        2. The top 3-5 most relevant books from the list above (by their index number)

                        3. A recommendation response that:
                        - Directly addresses the query
                        - Recommends ONE book from the top matches
                        - Explains why in 2-3 sentences
                        - Sounds natural (vary phrasing)

                        Output as JSON:
                        {{
                        "query": "user's natural question",
                        "retrieved_indices": [0, 2, 4],
                        "response": "Based on your interest in [query aspect], I recommend [book title] by [author]. [Explanation of why it fits]. [What makes it compelling]."
                        }}

                        IMPORTANT: Vary query complexity and response style. Don't repeat patterns."""
        
        response = await self.generate_content(prompt, format=OutputSchema)
        
        # Parse the JSON output from the response
        import json
        output_data = json.loads(response.output_text)
        parsed_output = OutputSchema(**output_data)
        return parsed_output

    async def generate_ner_example(self, books_sample):
        """Generate 2 new diverse, user-like NER examples using real book titles and authors."""
        # Extract titles and authors from the books sample
        books_info = []
        for _, row in books_sample.iterrows():
            books_info.append({
                "title": row['title'],
                "author": row['author']
            })
        
        prompt = f"""You are generating training data for a Named Entity Recognition (NER) system that identifies book titles and authors in natural user sentences.

Use these real books as source material:
{books_info}

Generate 2 NEW examples that sound like REAL users talking naturally about books. Be creative and diverse!

Vary the sentence patterns - use things like:
- Casual mentions: "My friend won't stop talking about Dune"
- Questions: "Who wrote The Hobbit again?"
- Recommendations: "You should totally check out 1984"
- Opinions: "I think Tolkien is overrated honestly"
- Partial recalls: "That book by the guy who wrote Harry Potter..."
- Typos/informal: "just finished reading game of throns lol"
- Multiple books: "I prefer Sanderson over Rothfuss"
- Context mentions: "Reading The Name of the Wind on my commute"
- Negative opinions: "Couldn't get into Pride and Prejudice"
- Comparisons: "It's like a mix of Dune and Foundation"

For each example, provide:
- tokenized_text: List of tokens (words and punctuation as separate tokens)
- ner: List of span objects with start_idx, end_idx (token positions, 0-indexed, inclusive), and label ("title" or "author")

IMPORTANT:
- Make sentences sound NATURAL and CONVERSATIONAL, not formal
- Tokenize properly: punctuation should be separate tokens
- Contractions like "don't" can be one token or split - be consistent
- Include typos, slang, or informal language sometimes
- Not every sentence needs both title AND author
- Use the ACTUAL book titles and author names from the list above

Output as JSON:
{{
  "items": [
    {{
      "tokenized_text": ["My", "friend", "keeps", "recommending", "Dune", ",", "is", "it", "good", "?"],
      "ner": [{{"start_idx": 4, "end_idx": 4, "label": "title"}}]
    }},
    {{
      "tokenized_text": ["Honestly", "I", "think", "Tolkien", "is", "a", "bit", "overrated"],
      "ner": [{{"start_idx": 3, "end_idx": 3, "label": "author"}}]
    }}
  ]
}}

Generate 2 diverse, natural-sounding examples now:"""
        
        response = await self.generate_content(prompt, format=NERList)
        
        # Parse the JSON output from the response
        import json
        output_data = json.loads(response.output_text)
        parsed_output = NERList(**output_data)
        return parsed_output