from pydantic import BaseModel, ConfigDict
from typing import List
import orjson

def orjson_dumps(v, *, default):
    return orjson.dumps(v, default=default).decode()

class OutputSchema(BaseModel):
    model_config = ConfigDict(json_dumps=orjson_dumps)

    query: str
    retrieved_indices: List[int]
    response: str


class TextClassificationSchema(BaseModel):
    model_config = ConfigDict(json_dumps=orjson_dumps)

    query: str
    genres: List[str]



