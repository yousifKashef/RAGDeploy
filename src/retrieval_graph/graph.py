import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain import hub

llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = ElasticsearchStore(
    es_url="https://3e3128b1cc10435eb0c19bdaa0fc4626.me-south-1.aws.elastic-cloud.com:9243",
    es_api_key=os.environ["ELASTICSEARCH_API_KEY"],
    index_name="procedure_test",
    embedding=embeddings
)



class RetrievalState(TypedDict):
    question: str
    company_tag: str
    context: List[Document]
    answer: str



def retrieve(state: RetrievalState):
    retrieved_docs = vector_store.similarity_search(
        query=state["question"],
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
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content
    })
    response = llm.invoke(messages)

    return {"answer": response.content}

# Build the Retrieval Graph
retrieval_graph_builder = StateGraph(RetrievalState).add_sequence([retrieve, generate])
retrieval_graph_builder.add_edge(START, "retrieve")
graph = retrieval_graph_builder.compile()
graph.name = "New Graph"