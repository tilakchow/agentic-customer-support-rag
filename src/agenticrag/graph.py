from typing import Any

from langgraph.graph import END, START, StateGraph

from agenticrag.node_meta import NodeName
from agenticrag.nodes import (
    answer_insufficient,
    classify_scope,
    condense_query,
    direct_answer,
    escalate_to_human,
    generate_from_kb,
    generate_from_web,
    grade_kb_evidence,
    grade_web_evidence,
    refuse_out_of_scope,
    retrieve_kb,
    rewrite_query,
    route_question,
    search_web,
)
from agenticrag.routing import (
    decide_after_kb_grade,
    decide_after_web_grade,
    route_after_router,
    route_after_scope,
)
from agenticrag.state import AgentState


def create_agent(checkpointer: Any = None):
    """Compile and return the LangGraph agent.

    ``checkpointer`` must be provided explicitly when persistence is required.
    If *None*, a fresh in-memory :class:`~langgraph.checkpoint.memory.MemorySaver`
    is used (suitable only for development / testing).
    """
    workflow = StateGraph(AgentState)

    # Add Nodes using NodeName enum
    workflow.add_node(NodeName.CLASSIFY_SCOPE.value, classify_scope)
    workflow.add_node(NodeName.REFUSE_OUT_OF_SCOPE.value, refuse_out_of_scope)
    workflow.add_node(NodeName.ESCALATE_TO_HUMAN.value, escalate_to_human)
    workflow.add_node(NodeName.ROUTE_QUESTION.value, route_question)
    workflow.add_node(NodeName.CONDENSE_QUERY.value, condense_query)
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
    workflow.add_edge(START, NodeName.CLASSIFY_SCOPE.value)

    workflow.add_conditional_edges(
        NodeName.CLASSIFY_SCOPE.value,
        route_after_scope,
        {
            NodeName.ROUTE_QUESTION.value: NodeName.ROUTE_QUESTION.value,
            NodeName.REFUSE_OUT_OF_SCOPE.value: NodeName.REFUSE_OUT_OF_SCOPE.value,
            NodeName.ESCALATE_TO_HUMAN.value: NodeName.ESCALATE_TO_HUMAN.value,
            NodeName.ANSWER_INSUFFICIENT.value: NodeName.ANSWER_INSUFFICIENT.value,
        },
    )

    workflow.add_conditional_edges(
        NodeName.ROUTE_QUESTION.value,
        route_after_router,
        {
            NodeName.CONDENSE_QUERY.value: NodeName.CONDENSE_QUERY.value,
            NodeName.DIRECT_ANSWER.value: NodeName.DIRECT_ANSWER.value,
        },
    )

    workflow.add_edge(NodeName.CONDENSE_QUERY.value, NodeName.RETRIEVE_KB.value)
    workflow.add_edge(NodeName.RETRIEVE_KB.value, NodeName.GRADE_KB_EVIDENCE.value)
    workflow.add_conditional_edges(
        NodeName.GRADE_KB_EVIDENCE.value,
        decide_after_kb_grade,
        {
            NodeName.GENERATE_FROM_KB.value: NodeName.GENERATE_FROM_KB.value,
            NodeName.SEARCH_WEB.value: NodeName.SEARCH_WEB.value,
            NodeName.ANSWER_INSUFFICIENT.value: NodeName.ANSWER_INSUFFICIENT.value,
        },
    )

    workflow.add_edge(NodeName.SEARCH_WEB.value, NodeName.GRADE_WEB_EVIDENCE.value)
    workflow.add_conditional_edges(
        NodeName.GRADE_WEB_EVIDENCE.value,
        decide_after_web_grade,
        {
            NodeName.GENERATE_FROM_WEB.value: NodeName.GENERATE_FROM_WEB.value,
            NodeName.REWRITE_QUERY.value: NodeName.REWRITE_QUERY.value,
            NodeName.ANSWER_INSUFFICIENT.value: NodeName.ANSWER_INSUFFICIENT.value,
        },
    )

    workflow.add_edge(NodeName.REWRITE_QUERY.value, NodeName.RETRIEVE_KB.value)

    workflow.add_edge(NodeName.GENERATE_FROM_KB.value, END)
    workflow.add_edge(NodeName.GENERATE_FROM_WEB.value, END)
    workflow.add_edge(NodeName.DIRECT_ANSWER.value, END)
    workflow.add_edge(NodeName.ANSWER_INSUFFICIENT.value, END)
    workflow.add_edge(NodeName.REFUSE_OUT_OF_SCOPE.value, END)
    workflow.add_edge(NodeName.ESCALATE_TO_HUMAN.value, END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
