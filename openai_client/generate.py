import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import APIError, RateLimitError, AuthenticationError
from openai.types.responses import Response
from openai.lib._parsing._responses import type_to_text_format_param
from .setup import Provider
from utils import logger
from .ouput_schema import OutputSchema, TextClassificationSchema


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

    async def generate_classification_example(self, books_sample):
        TARGET_GENRES = [
            "Romance", "Fantasy", "Young Adult", "Contemporary", 
            "Nonfiction", "Mystery", "Historical Fiction", "Classics"
        ]
        
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
                    
            prompt = f"""You are generating training data for a book recommendation system.

                        Given these books from a database:
                        {books_info}

                        The allowed genres for classification are:
                        {', '.join(TARGET_GENRES)}

                        Generate a realistic example with:
                        1. A user query (natural language, varied style) looking for books similar to the ones provided.
                        CRITICAL: The query must describe the plot, mood, atmosphere, or setting WITHOUT explicitly naming the genre.
                        - Bad: "I want a horror book."
                        - Good: "I want a story that keeps me up at night and makes me afraid of the dark."
                        - Bad: "Looking for a romance novel."
                        - Good: "I need a story about two people falling in love against all odds."
                        
                        2. A list of genres from the ALLOWED LIST above that are explicitly or implicitly mentioned in the generated query.
                        Select the most relevant genres (max 2-3) from the allowed list.

                        Output as JSON:
                        {{
                        "query": "user's natural question (implicit description)",
                        "genres": ["Fantasy", "Young Adult"]
                        }}

                        IMPORTANT: Vary query complexity. Do NOT use the genre names in the query text. Only use genres from the allowed list."""
        
        response = await self.generate_content(prompt, format=TextClassificationSchema)
        
        # Parse the JSON output from the response
        import json
        output_data = json.loads(response.output_text)
        parsed_output = TextClassificationSchema(**output_data)
        
        # Strict filtering: remove any genres not in the allowed list
        parsed_output.genres = [g for g in parsed_output.genres if g in TARGET_GENRES]
        
        return parsed_output