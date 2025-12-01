from pydantic import BaseModel, ConfigDict
from typing import List, Union
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


class NERSpan(BaseModel):
    start_idx: int
    end_idx: int
    label: str


class NERItem(BaseModel):
    model_config = ConfigDict(json_dumps=orjson_dumps)

    tokenized_text: List[str]
    ner: List[NERSpan]  # List of NER spans with start_idx, end_idx, label


class NERList(BaseModel):
    model_config = ConfigDict(json_dumps=orjson_dumps)

    items: List[NERItem]



