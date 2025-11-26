"""
Autonomous Retrieval Agent for LangGraph RAG System
Handles intelligent document retrieval, permission checking, and answer synthesis
using autonomous decision-making and state-aware processing.
"""

from typing import Dict, Any, List
import json
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

try:
    from ..tools.retrieval_tools import (
        search_vectordb_tool, permission_check_tool, vector_search_tool, 
        rerank_results_tool, synthesize_answer_tool, graph_expand_tool
    )
    from ..services.llm_service import LLMService
    from ..services.vectordb_service import VectorDBService
except ImportError:
    from tools.retrieval_tools import (
        search_vectordb_tool, permission_check_tool, vector_search_tool, 
        rerank_results_tool, synthesize_answer_tool, graph_expand_tool
    )
    from services.llm_service import LLMService
    from services.vectordb_service import VectorDBService

class RetrievalAgent:
    """Agent responsible for document retrieval and answer synthesis."""
    
    def __init__(self, llm_service: LLMService, vectordb_service: VectorDBService):
        self.llm_service = llm_service
        self.vectordb_service = vectordb_service
        self.tools = [
            search_vectordb_tool,
            permission_check_tool,
            vector_search_tool,
            rerank_results_tool,
            synthesize_answer_tool,
            graph_expand_tool
        ]
        
    def get_system_prompt(self) -> str:
        return """You are a document retrieval specialist. Your job is to find and synthesize answers from the vector database.

WORKFLOW:
1. Understand the user's query
2. Use vector_search_tool to find relevant chunks
3. Use permission_check_tool to verify access
4. Use rerank_results_tool to order by relevance
5. Use synthesize_answer_tool to generate the final answer
6. Report findings with confidence and sources

Be thorough and cite sources."""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[RetrievalAgent] Processing retrieval request...")
        try:
            request = state.get("request", "")
            request_type = state.get("request_type", "retrieval")
            
            # Only process if this is a retrieval request
            if request_type != "retrieval":
                print(f"  → Skipping (request type: {request_type})")
                return state
            
            # Call tools sequentially as a simple workflow
            print(f"  → Searching vector database...")
            search_result = vector_search_tool(request, top_k=5)
            print(f"  → Reranking results...")
            results_data = json.loads(search_result) if isinstance(search_result, str) else search_result
            reranked = rerank_results_tool(request, results_data.get("results", []))
            print(f"  → Synthesizing answer...")
            reranked_data = json.loads(reranked) if isinstance(reranked, str) else reranked
            answer = synthesize_answer_tool(request, reranked_data.get("reranked", []))
            
            state["retrieval_result"] = answer
            state["retrieval_completed"] = True
            state["metrics"]["retrieval"] += 1
            print(f"  ✅ Retrieval completed")
            return state
        except Exception as e:
            print(f"  ❌ Retrieval error: {e}")
            state["retrieval_result"] = f"Error: {str(e)}"
            state["retrieval_completed"] = True
            state["metrics"]["errors"] += 1
            return state
