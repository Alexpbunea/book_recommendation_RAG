import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import DefaultAioHttpClient, APIError, RateLimitError, AuthenticationError, OpenAI, AsyncOpenAI
from openai.types.responses import Response
from openai.lib._parsing._responses import parse_response, type_to_text_format_param
from dotenv import load_dotenv
from utils import logger

load_dotenv()

class Provider:
    def __init__(self):
        self.general_model : str = "gpt-5-nano"
        
    

    def name(self) -> str:
        return "Azure OpenAI"
    
    def setup(self):
        logger.info("Setting up Azure OpenAI provider...")
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("ENDPOINT")
        print(api_key, endpoint)
        

        if not api_key:
            logger.critical("Azure API key, endopoint or api version missing in environment variables.")
            raise ValueError("Missing Azure OpenAI API key, endpoint or api version")

        try:
            self.async_client = AsyncOpenAI(api_key=api_key,
                                            base_url=endpoint,
                                            http_client=DefaultAioHttpClient())
            logger.info("Azure provider setup complete.")
        except Exception as e:
            logger.critical("Azure OpenAI setup failed.", exc_info=True)
            raise
    
    async def close(self):
        try:
            await self.async_client.close()
            logger.info("Azure OpenAI async client closed successfully")
        except Exception as e:
            # Don't raise - this is cleanup, log and continue
            logger.warning(f"Error closing Azure OpenAI async client: {e}")