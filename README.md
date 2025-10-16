# Adaptive RAG System with LangGraph

An intelligent Retrieval-Augmented Generation (RAG) system that dynamically adapts its retrieval and generation strategies based on query complexity and document quality. Built with LangGraph and Google Gemini.

## 🎯 What Makes This "Adaptive"?

Unlike traditional RAG systems that follow a fixed pipeline, this system makes intelligent decisions at multiple checkpoints:

- **Smart Routing**: Automatically decides whether to search your local knowledge base or the web
- **Quality Control**: Grades retrieved documents for relevance before using them
- **Self-Correction**: Detects hallucinations and retries generation with better context
- **Answer Validation**: Ensures the final answer actually addresses the user's question

## 🏗️ Architecture

```
User Question
    ↓
[Router: Local KB or Web?]
    ↓                    ↓
[Vector Store]      [Web Search]
    ↓                    ↓
[Grade Documents] ───────┘
    ↓
[Are docs relevant?] ──No─→ [Web Search]
    ↓ Yes
[Generate Answer]
    ↓
[Check Hallucinations] ──Failed─→ [Regenerate]
    ↓ Passed
[Check Answer Quality] ──Poor─→ [Get More Docs]
    ↓ Good
[Return Answer]
```

## 📁 Project Structure

```
building-adaptive-rag/
├── src/
│   ├── workflow/              # Core workflow logic
│   │   ├── chains/            # LLM processing chains
│   │   │   ├── router.py              # Query routing logic
│   │   │   ├── retrieval_grader.py    # Document relevance grading
│   │   │   ├── hallucination_grader.py # Hallucination detection
│   │   │   ├── answer_grader.py       # Answer quality validation
│   │   │   └── generation.py          # Answer generation
│   │   ├── nodes/             # Workflow execution nodes
│   │   │   ├── retrieve.py            # Vector store retrieval
│   │   │   ├── grade_documents.py     # Document filtering
│   │   │   ├── web_search.py          # Web search integration
│   │   │   └── generate.py            # Answer generation
│   │   ├── state.py           # State management
│   │   ├── consts.py          # Node name constants
│   │   └── graph.py           # Workflow orchestration
│   ├── models/
│   │   └── model.py           # LLM and embedding models
│   └── cli/
│       └── main.py            # Interactive CLI
├── data/
│   └── ingestion.py           # Document ingestion & vector store
├── tests/
│   └── test_chains.py         # Comprehensive test suite
├── .env                       # Environment variables
├── main.py                    # Application entry point
├── requirements.txt           # Dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd building-adaptive-rag
```

2. **Install UV (if not already installed)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create virtual environment**
```bash
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

5. **Set up environment variables**

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional (for LangSmith tracing)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=adaptive-rag
```

**Get your API keys:**
- Google AI Studio: https://makersuite.google.com/app/apikey
- Tavily: https://tavily.com/ (free tier available)
- LangSmith (optional): https://smith.langchain.com/

### Running the System

**Interactive CLI:**
```bash
python main.py
```

Example session:
```
Adaptive RAG System
Type 'quit' to exit.

Question: What is agent memory?
Processing...
---ROUTE QUESTION---
---RETRIEVE---
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---GENERATE---
---CHECK HALLUCINATIONS---

Answer: Agent memory enables AI agents to maintain persistent states...
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

Tests cover:
- ✅ Document relevance grading (positive & negative cases)
- ✅ Hallucination detection
- ✅ Query routing (vectorstore vs web search)
- ✅ Generation chain functionality

## 🔧 Configuration

### Switching LLM Providers

The system currently uses Google Gemini, but you can easily switch to other providers:

**OpenAI:**
```python
# In src/models/model.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

**AWS Bedrock:**
```python
# In src/models/model.py
from langchain_aws import ChatBedrock

llm_model = ChatBedrock(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-west-2",
    temperature=0
)
```

### Customizing the Knowledge Base

By default, the system indexes content from specific URLs. To use your own documents:

**Option 1: Change URLs**
```python
# In data/ingestion.py
urls = [
    "https://your-domain.com/doc1",
    "https://your-domain.com/doc2",
]
```

**Option 2: Load local files**
```python
# In data/ingestion.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader

loader = PyPDFLoader("path/to/your/document.pdf")
docs = loader.load()
```

**Option 3: Delete existing vector store to rebuild**
```bash
rm -rf ./chroma_langchain_db
python main.py  # Will create new vector store
```

## 📊 How It Works

### 1. Query Routing
The system analyzes your question and decides the best information source:
- **Vectorstore**: For questions about agents, prompt engineering, adversarial attacks (based on indexed content)
- **Web Search**: For current events, general knowledge, or topics outside the knowledge base

### 2. Document Grading
Retrieved documents are strictly evaluated for relevance:
- Each document must **directly** address the question
- Irrelevant documents trigger web search for additional context
- Prevents poor-quality context from contaminating answers

### 3. Hallucination Detection
Generated answers are verified against source documents:
- Checks if claims are grounded in retrieved facts
- Triggers regeneration if hallucinations detected
- Ensures factual accuracy

### 4. Answer Quality Check
Final validation ensures the answer is useful:
- Verifies the answer actually addresses the question
- Triggers additional search if answer is incomplete
- Only returns high-quality responses

## 🎓 Key Concepts

### State Management
All workflow nodes share a common state object:
```python
{
    "question": "User's question",
    "documents": [retrieved_docs],
    "generation": "Generated answer",
    "web_search": False  # Control flag
}
```

### Conditional Edges
Decision points that route between nodes:
```python
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,  # Decision function
    {
        "websearch": WEBSEARCH,  # If docs are bad
        "generate": GENERATE     # If docs are good
    }
)
```

### Chains
Reusable LLM processing pipelines:
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": data})
```

## 🔍 Example Workflows

### Scenario 1: Question answered from local knowledge
```
Q: "What is prompt engineering?"
→ Router: vectorstore
→ Retrieve: 4 relevant docs
→ Grade: All relevant
→ Generate: High-quality answer
→ Validate: Grounded & useful
→ Return answer
```

### Scenario 2: Needs web search
```
Q: "What's the weather in Tokyo?"
→ Router: websearch (not in knowledge base)
→ Web Search: Fetch current weather
→ Generate: Answer from web results
→ Validate: Grounded & useful
→ Return answer
```

### Scenario 3: Self-correction flow
```
Q: "What is agent memory?"
→ Router: vectorstore
→ Retrieve: 4 docs (2 relevant, 2 not)
→ Grade: Mixed quality → trigger web search
→ Web Search: Get additional context
→ Generate: Answer from local + web
→ Validate: Initially poor → regenerate
→ Validate: Now good
→ Return answer
```

## 🛠️ Troubleshooting

**Issue: `chroma_langchain_db` not found**
```bash
# Delete and rebuild
rm -rf ./chroma_langchain_db
python main.py
```

**Issue: API key errors**
```bash
# Verify .env file exists and contains valid keys
cat .env
```

**Issue: Tests failing**
```bash
# Ensure vector store is created first
python main.py  # Run once to create vector store
python -m pytest tests/ -v
```

**Issue: Slow performance**
- Reduce `max_results` in `web_search.py` (currently 3)
- Use a faster embedding model
- Implement caching for repeated queries


**Optimization tips:**
1. Use embedding-based document grading instead of LLM grading
2. Cache frequently asked questions
3. Batch document grading when possible
4. Use cheaper models for grading, premium models for generation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for more document types (PDF, DOCX, etc.)
- [ ] Implement caching layer for repeated queries
- [ ] Add streaming support for real-time responses
- [ ] Create web UI interface
- [ ] Add conversation memory for multi-turn dialogues
- [ ] Implement query expansion for better retrieval
- [ ] Add evaluation metrics (faithfulness, answer relevance, etc.)

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by [Marco's implementation](https://github.com/marco-ostaska/agentic-rag-langgraph)
- Based on research from [Adaptive RAG paper](https://arxiv.org/abs/2403.14403)
- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Adaptive RAG Blog Post](https://ai.plainenglish.io/building-agentic-rag-with-langgraph-mastering-adaptive-rag-for-production-c2c4578c836a)
- [RAG Best Practices](https://www.anthropic.com/research/building-effective-agents)

---

**Questions or issues?** Open an issue on GitHub or reach out to the maintainers.