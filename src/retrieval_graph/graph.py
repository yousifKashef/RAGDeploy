import os
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain import hub
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = ElasticsearchStore(
    es_url="https://my-elasticsearch-project-da8b8c.es.us-east-1.aws.elastic.cloud:443",
    es_api_key=os.environ["ELASTICSEARCH_API_KEY"],
    index_name="procedure_test",
    embedding=embeddings
)

class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company_tag: str


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=6, fetch_k=150)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke([SystemMessage("You are smart RAG bot. You get to decide when to answer and when to retrieve. You are currently deployed as a procedural bot. You answer questions about company procedures. So things like what do I do if always warrant a tool call to your retriever")] + state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to provide "
        "A concise but detailed answers to the question. Include relevant "
        "documentation and information, quoting them in a formatted "
        "manner to ensure clarity and traceability. "
        "\n\n"
        f"{docs_content}"
        "\n\n"
        "For quotations, use the following template: \n"
        "```quote\n"
        "{quote}\n"
        "```\n"
        "Ensure the answer is comprehensive and thoroughly explains "
        "the context and rationale behind it."
        "no need to format other text for appearance. so no titles in **"
        "if your answer involves the name of the form, at the very end after some white space, provide the name of the form within ###form name###. provide just this at the end nothing else. for example ###F.21234###"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)