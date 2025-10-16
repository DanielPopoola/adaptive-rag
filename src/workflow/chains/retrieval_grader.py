from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from models.model import llm_model


class GradeDocuments(BaseModel):
    
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


llm = llm_model
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a strict grader assessing relevance of a retrieved document to a user question.

The document must DIRECTLY address the user's question to be considered relevant.
Do NOT grade as relevant if:
- The document only shares a general topic area
- There is only tangential or indirect connection
- The document cannot help answer the specific question asked

Only grade as relevant if the document contains information that directly helps answer the question.

Give a binary score 'yes' or 'no'."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader