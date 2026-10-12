from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agenticrag.nodes import (
    NodeName,
    AgentState,
    route_question, route_after_router,
    retrieve_kb, grade_kb_evidence, decide_after_kb_grade,
    search_web, grade_web_evidence, decide_after_web_grade,
    rewrite_query, generate_from_kb, generate_from_web,
    direct_answer, answer_insufficient
)

_compiled_agent = None


def create_agent():
    workflow = StateGraph(AgentState)

    # Add Nodes using NodeName enum
    workflow.add_node(NodeName.ROUTE_QUESTION.value, route_question)
    workflow.add_node(NodeName.RETRIEVE_KB.value, retrieve_kb)
    workflow.add_node(NodeName.GRADE_KB_EVIDENCE.value, grade_kb_evidence)
    workflow.add_node(NodeName.SEARCH_WEB.value, search_web)
    workflow.add_node(NodeName.GRADE_WEB_EVIDENCE.value, grade_web_evidence)
    workflow.add_node(NodeName.REWRITE_QUERY.value, rewrite_query)
    workflow.add_node(NodeName.GENERATE_FROM_KB.value, generate_from_kb)
    workflow.add_node(NodeName.GENERATE_FROM_WEB.value, generate_from_web)
    workflow.add_node(NodeName.DIRECT_ANSWER.value, direct_answer)
    workflow.add_node(NodeName.ANSWER_INSUFFICIENT.value, answer_insufficient)

    # Add Edges
    workflow.add_edge(START, NodeName.ROUTE_QUESTION.value)

    workflow.add_conditional_edges(
        NodeName.ROUTE_QUESTION.value,
        route_after_router,
        {
            NodeName.RETRIEVE_KB.value: NodeName.RETRIEVE_KB.value,
            NodeName.DIRECT_ANSWER.value: NodeName.DIRECT_ANSWER.value,
        }
    )

    workflow.add_edge(NodeName.RETRIEVE_KB.value, NodeName.GRADE_KB_EVIDENCE.value)
    workflow.add_conditional_edges(
        NodeName.GRADE_KB_EVIDENCE.value,
        decide_after_kb_grade,
        {
            NodeName.GENERATE_FROM_KB.value: NodeName.GENERATE_FROM_KB.value,
            NodeName.SEARCH_WEB.value: NodeName.SEARCH_WEB.value,
            NodeName.ANSWER_INSUFFICIENT.value: NodeName.ANSWER_INSUFFICIENT.value,
        }
    )

    workflow.add_edge(NodeName.SEARCH_WEB.value, NodeName.GRADE_WEB_EVIDENCE.value)
    workflow.add_conditional_edges(
        NodeName.GRADE_WEB_EVIDENCE.value,
        decide_after_web_grade,
        {
            NodeName.GENERATE_FROM_WEB.value: NodeName.GENERATE_FROM_WEB.value,
            NodeName.REWRITE_QUERY.value: NodeName.REWRITE_QUERY.value,
            NodeName.ANSWER_INSUFFICIENT.value: NodeName.ANSWER_INSUFFICIENT.value,
        }
    )

    workflow.add_edge(NodeName.REWRITE_QUERY.value, NodeName.RETRIEVE_KB.value)
    
    workflow.add_edge(NodeName.GENERATE_FROM_KB.value, END)
    workflow.add_edge(NodeName.GENERATE_FROM_WEB.value, END)
    workflow.add_edge(NodeName.DIRECT_ANSWER.value, END)
    workflow.add_edge(NodeName.ANSWER_INSUFFICIENT.value, END)

    # Memory checkpointer to remember chat history
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def get_agent_app():
    """Lazy getter for compiled agent graph."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = create_agent()
    return _compiled_agent
