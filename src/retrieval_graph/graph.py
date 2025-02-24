import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain import hub
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts.chat import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = ElasticsearchStore(
    es_url="https://3e3128b1cc10435eb0c19bdaa0fc4626.me-south-1.aws.elastic-cloud.com:9243",
    es_api_key=os.environ["ELASTICSEARCH_API_KEY"],
    index_name="procedure_test",
    embedding=embeddings
)



class RetrievalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company_tag: str
    context: List[Document]
    answer: str


def retrieve(state: RetrievalState):
    query = ""
    for doc in state["context"]:
        query += doc.page_content
    query += state["messages"][-1].content
    retrieved_docs = vector_store.similarity_search(
        query=query,
        filter=[{"term": {"metadata.company_tag.keyword": state["company_tag"]}}]
    )
    return {"context": retrieved_docs}

def generate(state: RetrievalState):
    for doc in state["context"]:
        doc_name = doc.metadata.get("file_name", "Unknown Source")
        chunk_preview = doc.page_content[:200]
        print(f"--- Document: {doc_name} ---")
        print(f"Chunk Preview: {chunk_preview}...\n")

    # Format documents into a single context string
    docs_content = "\n\n".join(
        f"[{doc.metadata.get('file_name', 'Unknown Source')}]\n{doc.page_content}"
        for doc in state["context"]
    )

    # Construct the prompt and invoke LLM
    prompt = template = ChatPromptTemplate([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("system", "{context}"),
    ("human", "{messages}"),
])
    messages = prompt.invoke({
        "messages": state["messages"],
        "context": docs_content
    })
    response = llm.invoke(messages)

    return {"answer": response.content}

# Build the Retrieval Graph
retrieval_graph_builder = StateGraph(RetrievalState).add_sequence([retrieve, generate])
retrieval_graph_builder.add_edge(START, "retrieve")
graph = retrieval_graph_builder.compile()
graph.name = "New Graph"