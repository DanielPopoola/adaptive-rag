from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")