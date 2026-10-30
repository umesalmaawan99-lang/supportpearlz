import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
from src.rag.retriever import get_retriever

logger = logging.getLogger("SupportPearlz.RAG.Chain")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Builds and returns the full RAG chain using OpenAI and FAISS retriever."""
    retriever = get_retriever()
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        openai_api_key=settings.openai_api_key
    )
    
    # Prompt template for customer support with fallback guardrail
    template = """You are an official, helpful, and professional customer support assistant for "SupportPearlz". 
Answer the user's question accurately using ONLY the provided context below. 
If the answer cannot be found within the context, politely inform the user that you do not have that information and suggest they contact support at support@supportpearlz.com.

Context:
{context}

User Question: {question}

Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Construct LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain successfully constructed.")
    return rag_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    query = "What is the return policy for digital subscriptions?"
    response = chain.invoke(query)
    print(f"\nQuery: {query}")
    print(f"Answer:\n{response}")