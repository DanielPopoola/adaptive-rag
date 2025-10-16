from pprint import pprint
import pytest
from dotenv import load_dotenv

load_dotenv()

from workflow.chains.generation import generation_chain
from workflow.chains.hallucination_grader import (GradeHallucinations, hallucination_grader)
from workflow.chains.retrieval_grader import GradeDocuments, retrieval_grader
from workflow.chains.router import RouteQuery, question_router
from data.ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "What are the key components of an LLM-powered autonomous agent system?"
    docs = retriever.invoke(question)
    
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "What is retrieval augmented generation?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to cook pasta with mushrooms", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "How do language models work?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "What are the benefits of vector databases?"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "What are the benefits of vector databases?"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "To bake a perfect chocolate cake, you need to preheat the oven to 350 degrees",
        }
    )
    assert res.binary_score == 'no'


def test_router_to_vectorstore() -> None:
    question = "What is the difference between RAG and fine-tuning?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "What is the current weather in Tokyo?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"