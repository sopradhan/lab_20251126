"""
LangGraph Autonomous Agentic RAG System with Reinforcement Learning
Multi-agent system orchestrator that coordinates ingestion, retrieval, and hallucination detection
using LangGraph for autonomous decision-making and reinforcement learning for optimization.
"""

from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import json
import time
from pathlib import Path
import sys
import os

# Setup imports to handle both relative and absolute paths
def _setup_imports():
    """Setup module paths for imports"""
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

_setup_imports()

# Try relative imports first, fall back to direct imports
try:
    from .agents.ingestion_agent import IngestionAgent
    from .agents.retrieval_agent import RetrievalAgent
    from .agents.hallucination_agent import HallucinationAgent
    from .services.llm_service import LLMService
    from .services.vectordb_service import VectorDBService
    from .config.env_config import EnvConfig
except ImportError:
    try:
        from agents.ingestion_agent import IngestionAgent
        from agents.retrieval_agent import RetrievalAgent
        from agents.hallucination_agent import HallucinationAgent
        from services.llm_service import LLMService
        from services.vectordb_service import VectorDBService
        from config.env_config import EnvConfig
    except ImportError as e:
        print(f"Warning: Could not import from agents/services/config: {e}")
        raise

class RAGState(TypedDict):
    request: str
    request_type: Literal["ingestion", "retrieval"]
    ingestion_result: str
    retrieval_result: str
    hallucination_suggestions: list
    metrics: Dict[str, Any]
    ingestion_completed: bool
    retrieval_completed: bool
    hallucination_completed: bool

class LangGraphRAGOrchestrator:
    """
    Autonomous Agentic RAG System Orchestrator using LangGraph with Reinforcement Learning
    
    This orchestrator coordinates multiple specialized agents:
    - Ingestion Agent: Handles document processing and storage
    - Retrieval Agent: Manages vector search and answer synthesis  
    - Hallucination Agent: Uses Q-learning for hallucination detection and optimization
    
    The system uses LangGraph for state management and autonomous agent coordination,
    implementing reinforcement learning for continuous improvement of retrieval quality.
    """
    
    def __init__(self):
        # Initialize services
        config_path = Path(__file__).parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        
        self.llm_service = LLMService(llm_config)
        db_path = EnvConfig.get_chroma_db_path()
        self.vectordb_service = VectorDBService(persist_directory=db_path)
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent(self.llm_service, self.vectordb_service)
        self.retrieval_agent = RetrievalAgent(self.llm_service, self.vectordb_service)
        self.hallucination_agent = HallucinationAgent()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Initial metrics
        self.metrics = {
            "operations": 0,
            "ingestion": 0,
            "retrieval": 0,            "errors": 0,
            "tokens": 0,
            "embeddings": 0,
            "hallucination_checks": 0
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the autonomous LangGraph workflow with RL optimization."""
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("router", self._route_request)
        graph.add_node("ingestion", self.ingestion_agent.process)
        graph.add_node("retrieval", self.retrieval_agent.process)
        graph.add_node("hallucination", self._autonomous_hallucination_node)
        
        # Add edges
        graph.add_conditional_edges(
            START,
            self._route_request,
            {
                "ingestion": "ingestion",
                "retrieval": "retrieval"
            }
        )
        
        graph.add_edge("ingestion", "hallucination")
        graph.add_edge("retrieval", "hallucination")
        graph.add_edge("hallucination", END)
        
        return graph.compile()

    def _autonomous_hallucination_node(self, state: RAGState) -> RAGState:
        """Enhanced autonomous hallucination detection with RL."""
        print("[Orchestrator] Running autonomous hallucination agent with RL...")
        
        # Add processing time for RL state computation
        start_time = time.time()
        
        # Use the autonomous processing method
        if hasattr(self.hallucination_agent, 'autonomous_process'):
            state = self.hallucination_agent.autonomous_process(state)
        else:
            state = self.hallucination_agent.process(state)
        
        # Record processing time for metrics
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        state["processing_time"] = processing_time
        
        # Update orchestrator metrics
        self.metrics["hallucination_checks"] += 1
        if "autonomous_metrics" in state:
            print(f"  → RL Action: {state['autonomous_metrics'].get('action', 'unknown')}")
            print(f"  → RL Reward: {state['autonomous_metrics'].get('reward', 0.0):.3f}")
        
        return state
    
    def _route_request(self, state: RAGState) -> Literal["ingestion", "retrieval"]:
        """Autonomous intelligent routing using LLM decision-making."""
        request = state["request"]
        print(f"[Router] Autonomous analysis of request: '{request[:100]}...'")
        
        # Use LLM to make intelligent routing decision
        routing_prompt = f"""You are an autonomous routing agent in a RAG system. Analyze this user request and decide whether it's for:
1. INGESTION: Adding, storing, importing, uploading documents/data to the system
2. RETRIEVAL: Searching, querying, finding information from existing data

User request: "{request}"

Respond with ONLY one word: "ingestion" or "retrieval"
Be intelligent about your decision - consider context, intent, and action verbs."""

        try:
            # Get LLM decision for routing
            response = self.llm_service.generate_response(routing_prompt)
            decision = response.strip().lower()
            
            if "ingestion" in decision:
                print(f"[Router] LLM Decision: INGESTION")
                state["request_type"] = "ingestion"
                return "ingestion"
            elif "retrieval" in decision:
                print(f"[Router] LLM Decision: RETRIEVAL")
                state["request_type"] = "retrieval"
                return "retrieval"
            else:
                # Fallback to context analysis if LLM response is unclear
                return self._fallback_routing(state, request)
                
        except Exception as e:
            print(f"[Router] LLM routing failed: {e}, using fallback")
            return self._fallback_routing(state, request)
    
    def _fallback_routing(self, state: RAGState, request: str) -> Literal["ingestion", "retrieval"]:
        """Fallback routing with contextual intelligence."""
        request_lower = request.lower()
        
        # Intelligent pattern matching with context
        ingestion_patterns = {
            "action_verbs": ["ingest", "upload", "add", "import", "store", "load", "process", "save"],
            "file_indicators": [".txt", ".pdf", ".doc", ".csv", ".json", "file", "document"],
            "data_indicators": ["table", "database", "sqlite", "data", "records", "rows"],
            "path_indicators": ["path", "folder", "directory", "location"]
        }
        
        retrieval_patterns = {
            "question_words": ["what", "how", "when", "where", "why", "which", "who"],
            "search_verbs": ["find", "search", "look", "query", "get", "retrieve", "show"],
            "info_requests": ["information", "details", "data", "content", "results"],
            "question_marks": ["?"]
        }
        
        # Score-based decision making
        ingestion_score = 0
        retrieval_score = 0
        
        for category, patterns in ingestion_patterns.items():
            for pattern in patterns:
                if pattern in request_lower:
                    weight = 2 if category == "action_verbs" else 1
                    ingestion_score += weight
        
        for category, patterns in retrieval_patterns.items():
            for pattern in patterns:
                if pattern in request_lower:
                    weight = 2 if category == "question_words" else 1
                    retrieval_score += weight
        
        # Make intelligent decision
        if ingestion_score > retrieval_score:
            print(f"[Router] Context Analysis: INGESTION (score: {ingestion_score} vs {retrieval_score})")
            state["request_type"] = "ingestion"
            return "ingestion"
        elif retrieval_score > ingestion_score:
            print(f"[Router] Context Analysis: RETRIEVAL (score: {retrieval_score} vs {ingestion_score})")
            state["request_type"] = "retrieval"
            return "retrieval"
        else:
            # Tie-breaker: analyze sentence structure
            if request.strip().endswith("?") or any(word in request_lower for word in ["what", "how", "tell me"]):
                print(f"[Router] Tie-breaker: Question detected → RETRIEVAL")
                state["request_type"] = "retrieval"
                return "retrieval"
            else:
                print(f"[Router] Tie-breaker: Action assumed → INGESTION")
                state["request_type"] = "ingestion"
                return "ingestion"
        
    def process_request(self, request: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a RAG request through the autonomous graph with optional parameters.
        
        Args:
            request: The user request string
            params: Optional parameters that can influence agent behavior
        """
        # Enhanced state with autonomous capabilities
        initial_state = RAGState(
            request=request,
            request_type="unknown",  # Let autonomous router determine this
            ingestion_result="",
            retrieval_result="",
            hallucination_suggestions=[],
            metrics=self.metrics.copy(),
            ingestion_completed=False,
            retrieval_completed=False,
            hallucination_completed=False
        )
        
        # Add optional parameters to state for autonomous agent decision-making
        if params:
            initial_state["agent_params"] = params
            print(f"[Orchestrator] Processing with autonomous parameters: {params}")
        
        # Add autonomous context analysis
        initial_state["autonomous_context"] = self._analyze_request_context(request)
        
        print(f"[Orchestrator] Starting autonomous processing pipeline...")
        result = self.graph.invoke(initial_state)
        
        # Update global metrics
        self.metrics.update(result["metrics"])
        self.metrics["operations"] += 1
        
        # Autonomous response formatting
        formatted_result = self._format_autonomous_response(result)
        
        return formatted_result
    
    def _analyze_request_context(self, request: str) -> Dict[str, Any]:
        """Analyze request context for autonomous agent decision-making."""
        context = {
            "request_length": len(request),
            "has_file_path": any(ext in request.lower() for ext in ['.txt', '.pdf', '.csv', '.json', '.doc']),
            "has_table_reference": any(word in request.lower() for word in ['table', 'database', 'sqlite', 'records']),
            "is_question": request.strip().endswith('?'),
            "question_words": sum(1 for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who'] if word in request.lower()),
            "action_words": sum(1 for word in ['ingest', 'add', 'import', 'process', 'analyze', 'store'] if word in request.lower()),
            "urgency_indicators": sum(1 for word in ['urgent', 'asap', 'immediately', 'quickly', 'emergency'] if word in request.lower()),
            "complexity_score": len(request.split()) / 10.0  # Simple complexity measure
        }
        
        # Autonomous intent detection
        if context["action_words"] > context["question_words"]:
            context["likely_intent"] = "action"
        elif context["question_words"] > 0 or context["is_question"]:
            context["likely_intent"] = "query"
        else:
            context["likely_intent"] = "mixed"
            
        return context
    
    def _format_autonomous_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format response with autonomous system metadata."""
        formatted = {
            "result": result.get("ingestion_result", "") or result.get("retrieval_result", ""),
            "suggestions": result.get("hallucination_suggestions", []),
            "metrics": result["metrics"],
            "autonomous_metadata": {
                "processing_type": result.get("request_type", "unknown"),
                "processing_time": result.get("processing_time", 0),
                "rl_action_taken": result.get("autonomous_metrics", {}).get("action", "none"),
                "rl_confidence": result.get("autonomous_metrics", {}).get("reward", 0.0),
                "context_analysis": result.get("autonomous_context", {})
            }
        }
        
        return formatted
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return self.hallucination_agent.get_metrics_report()
    
    def train_hallucination_agent(self, timesteps: int = None, adaptive: bool = True):
        """
        Train the RL-based hallucination agent with autonomous adaptation.
        
        Args:
            timesteps: Number of training steps (auto-determined if None)
            adaptive: Whether to use adaptive training based on system performance
        """
        if timesteps is None:
            # Autonomous determination of training steps based on system state
            base_steps = 5000
            performance_factor = max(0.5, min(2.0, 1.0 - (self.metrics.get("errors", 0) / max(1, self.metrics.get("operations", 1)))))
            timesteps = int(base_steps * performance_factor)
            print(f"[Training] Autonomously determined training steps: {timesteps} (performance factor: {performance_factor:.2f})")
        
        if adaptive:
            print(f"[Training] Using adaptive RL training with {timesteps} steps...")
            # Adaptive training that adjusts based on current performance
            if hasattr(self.hallucination_agent, 'train_adaptive'):
                self.hallucination_agent.train_adaptive(timesteps, self.metrics)
            else:
                self.hallucination_agent.train(timesteps)
        else:
            print(f"[Training] Using standard RL training with {timesteps} steps...")
            self.hallucination_agent.train(timesteps)
    
    def autonomous_optimize(self) -> Dict[str, Any]:
        """
        Autonomous system optimization based on performance metrics.
        
        Returns:
            Dict containing optimization results and recommendations
        """
        print("[Optimizer] Running autonomous system optimization...")
        
        current_metrics = self.metrics
        optimization_results = {
            "recommendations": [],
            "actions_taken": [],
            "performance_score": 0.0
        }
        
        # Calculate performance score
        total_ops = max(1, current_metrics.get("operations", 1))
        error_rate = current_metrics.get("errors", 0) / total_ops
        success_rate = 1.0 - error_rate
        
        optimization_results["performance_score"] = success_rate
        
        # Autonomous optimization decisions
        if error_rate > 0.1:  # More than 10% error rate
            optimization_results["recommendations"].append("High error rate detected - consider retraining RL agent")
            if error_rate > 0.2:  # Critical error rate
                print("[Optimizer] Critical error rate - initiating autonomous retraining...")
                self.train_hallucination_agent(adaptive=True)
                optimization_results["actions_taken"].append("Autonomous RL retraining initiated")
        
        if current_metrics.get("hallucination_checks", 0) < total_ops * 0.5:
            optimization_results["recommendations"].append("Low hallucination check coverage - consider increasing RL agent engagement")
        
        # Memory optimization
        if total_ops > 1000:
            print("[Optimizer] High operation count - optimizing memory usage...")
            optimization_results["actions_taken"].append("Memory optimization applied")
            # Reset some accumulated metrics to prevent memory bloat
            self.metrics = {key: min(val, 10000) if isinstance(val, (int, float)) else val 
                          for key, val in self.metrics.items()}
        
        return optimization_results

if __name__ == "__main__":
    # Autonomous Agentic RAG System - Dynamic Example Usage
    print("🤖 Initializing Autonomous Agentic RAG System with Reinforcement Learning")
    orchestrator = LangGraphRAGOrchestrator()
    
    # Train the RL agent for optimal performance
    print("🧠 Training hallucination detection agent with reinforcement learning...")
    orchestrator.train_hallucination_agent(3000)
    
    # Autonomous examples - let the LLM agents decide what to do
    print("\n" + "="*60)
    print("🔄 AUTONOMOUS PROCESSING EXAMPLES")
    print("="*60)
    
    # Example 1: Autonomous table ingestion
    autonomous_request1 = "I need to analyze incident data from our database. Can you help me ingest the incidents table and prepare it for analysis?"
    print(f"\n📝 User Request: {autonomous_request1}")
    result1 = orchestrator.process_request(autonomous_request1)
    print(f"🔄 Autonomous System Response: {result1['result'][:200]}...")
    print(f"📊 Suggestions: {result1.get('suggestions', [])}")
    
    # Example 2: Autonomous document ingestion with path detection
    autonomous_request2 = "There's a important document located at data/incident_reports.txt that needs to be processed for our knowledge base"
    print(f"\n📝 User Request: {autonomous_request2}")
    result2 = orchestrator.process_request(autonomous_request2)
    print(f"🔄 Autonomous System Response: {result2['result'][:200]}...")
    
    # Example 3: Autonomous intelligent query
    autonomous_request3 = "What patterns can you identify in the incident data? I'm particularly interested in severity trends and common categories."
    print(f"\n📝 User Request: {autonomous_request3}")
    result3 = orchestrator.process_request(autonomous_request3)
    print(f"🔄 Autonomous System Response: {result3['result'][:200]}...")
    print(f"💡 RL Suggestions: {result3.get('suggestions', [])}")
    
    # Example 4: Mixed intent - let the LLM decide
    autonomous_request4 = "I have a CSV file with customer feedback data - could you analyze it and tell me the main themes?"
    print(f"\n📝 User Request: {autonomous_request4}")
    result4 = orchestrator.process_request(autonomous_request4)
    print(f"🔄 Autonomous System Response: {result4['result'][:200]}...")
    
    # Get comprehensive metrics from the autonomous system
    print("\n" + "="*60)
    print("📈 AUTONOMOUS SYSTEM METRICS")
    print("="*60)
    metrics = orchestrator.get_metrics_report()
    print(f"🔢 Operations: {orchestrator.metrics.get('operations', 0)}")
    print(f"📥 Ingestions: {orchestrator.metrics.get('ingestion', 0)}")
    print(f"🔍 Retrievals: {orchestrator.metrics.get('retrieval', 0)}")
    print(f"🧠 RL Checks: {orchestrator.metrics.get('hallucination_checks', 0)}")
    print(f"⚠️  Errors: {orchestrator.metrics.get('errors', 0)}")
    if metrics:
        print(f"🎯 RL Performance: {metrics.get('rl_metrics', {})}")
    
    print("\n🎉 Autonomous Agentic RAG System demonstration complete!")
