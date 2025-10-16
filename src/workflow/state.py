from typing import List, TypedDict
from langchain.schema import Document

class GraphState(TypedDict):
    """State object for workflow containing query, documents, and control flags."""
    question: str
    generation: str
    web_search: bool
    documents: List[Document]