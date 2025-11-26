import numpy as np
from typing import Dict, Any, List, Optional
import json
import time
import uuid

# --- Enhanced Autonomous Agent Components ---
class AutonomousRewardCalculator:
    """Calculate rewards for reinforcement learning based on retrieval quality."""
    
    def __init__(self):
        self.quality_weights = {
            'similarity': 0.3,
            'relevance': 0.25,
            'diversity': 0.2,
            'latency': 0.15,
            'accuracy': 0.1
        }
    
    def calculate_reward(self, state_metrics: Dict[str, Any], action_taken: str) -> float:
        """Calculate reward based on action effectiveness and retrieval quality."""
        reward = 0.0
        
        # Similarity reward
        if 'similarities' in state_metrics:
            similarities = state_metrics['similarities']
            avg_sim = np.mean(similarities) if similarities else 0.0
            reward += self.quality_weights['similarity'] * avg_sim
        
        # Latency penalty (lower latency = higher reward)
        if 'latency' in state_metrics:
            latency_score = max(0, 1 - (state_metrics['latency'] / 5000))  # normalize to 5s max
            reward += self.quality_weights['latency'] * latency_score
        
        # Action-specific bonuses
        action_bonuses = {
            'vector_search': 0.1,
            'rerank': 0.15,
            'expand': 0.05,
            'hybrid': 0.2,
            'cache': 0.25
        }
        reward += action_bonuses.get(action_taken, 0.0)
        
        return reward

class AutonomousStateManager:
    """Manages state transitions and learning for autonomous decision-making."""
    
    def __init__(self):
        self.state_history = []
        self.performance_metrics = {
            'avg_latency': [],
            'avg_similarity': [],
            'action_effectiveness': {}
        }
    
    def update_performance(self, state: np.ndarray, action: str, reward: float):
        """Update performance tracking for autonomous learning."""
        self.state_history.append({
            'timestamp': time.time(),
            'state': state.tolist(),
            'action': action,
            'reward': reward,
            'session_id': str(uuid.uuid4())[:8]
        })
        
        # Track action effectiveness
        if action not in self.performance_metrics['action_effectiveness']:
            self.performance_metrics['action_effectiveness'][action] = []
        self.performance_metrics['action_effectiveness'][action].append(reward)
    
    def get_best_action_for_state(self, state: np.ndarray) -> Optional[str]:
        """Autonomous action recommendation based on historical performance."""
        if not self.state_history:
            return None
        
        # Find similar states and their best actions
        similar_states = []
        for record in self.state_history:
            historical_state = np.array(record['state'])
            similarity = np.dot(state, historical_state) / (np.linalg.norm(state) * np.linalg.norm(historical_state))
            if similarity > 0.8:  # High similarity threshold
                similar_states.append((record['action'], record['reward'], similarity))
        
        if not similar_states:
            return None
        
        # Return action with highest weighted reward
        best_action = max(similar_states, key=lambda x: x[1] * x[2])
        return best_action[0]

# --- State Representation ---
def compute_state_vector(query_embedding, chunk_embeddings, token_count, latency, chunk_lengths, complexity_score, chunk_similarities):
    # Embedding statistics
    similarities = np.array(chunk_similarities)
    mean_sim = float(np.mean(similarities)) if len(similarities) else 0.0
    std_sim = float(np.std(similarities)) if len(similarities) else 0.0
    max_sim = float(np.max(similarities)) if len(similarities) else 0.0
    min_sim = float(np.min(similarities)) if len(similarities) else 0.0
    query_norm = float(np.linalg.norm(query_embedding)) if query_embedding is not None else 0.0
    chunk_norms = [float(np.linalg.norm(e)) for e in chunk_embeddings] if chunk_embeddings else [0.0]
    std_chunk_norm = float(np.std(chunk_norms)) if chunk_norms else 0.0
    # Metadata features
    norm_token_count = min(1.0, token_count/50000)
    norm_latency = min(1.0, latency/5000)
    norm_chunk_count = min(1.0, len(chunk_embeddings)/100)
    avg_chunk_len = float(np.mean(chunk_lengths)) if chunk_lengths else 0.0
    norm_avg_chunk_len = min(1.0, avg_chunk_len/10000)
    # 11-dim state
    return np.array([
        mean_sim, std_sim, max_sim, min_sim, query_norm, std_chunk_norm,
        norm_token_count, complexity_score, norm_latency, norm_chunk_count, norm_avg_chunk_len
    ], dtype=np.float32)

# --- Q-Learning Agent ---
class QLearningAgent:
    ACTIONS = ["vector_search", "rerank", "expand", "hybrid", "cache"]
    def __init__(self, state_dim=11, n_actions=5, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # key: tuple(state.round(2)), value: np.array(n_actions)
        self.last_state = None
        self.last_action = None
    def _state_key(self, state):
        return tuple(np.round(state, 2))
    def select_action(self, state):
        key = self._state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.q_table[key]))
        self.last_state = key
        self.last_action = action
        return action
    def update(self, state, reward):
        key = self._state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        prev_q = self.q_table[self.last_state][self.last_action]
        max_next_q = np.max(self.q_table[key])
        self.q_table[self.last_state][self.last_action] = prev_q + self.alpha * (reward + self.gamma * max_next_q - prev_q)

# --- Reward Calculation ---
def calculate_reward(top5_sim, latency, tokens, user_feedback):
    relevance = float(np.mean(top5_sim)) if top5_sim else 0.0
    speed = max(0, 1 - latency/5000)
    efficiency = max(0, 1 - tokens/50000)
    feedback = user_feedback if user_feedback is not None else 0.5
    return 0.5*relevance + 0.2*speed + 0.15*efficiency + 0.15*feedback

# --- HallucinationAgent ---
class HallucinationAgent:
    """
    Autonomous Hallucination Detection Agent with Advanced Reinforcement Learning
    
    This agent uses Q-learning and autonomous state management to continuously
    improve retrieval quality and detect potential hallucinations in RAG responses.
    """
    
    def __init__(self):
        self.q_agent = QLearningAgent()
        self.reward_calculator = AutonomousRewardCalculator()
        self.state_manager = AutonomousStateManager()
        self.metrics_history = []
        self.autonomous_mode = True  # Enable autonomous decision-making

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Analyze query (HITL loop: user_feedback should be set externally, e.g. by a human-in-the-loop UI or feedback system)
        # 2. Gather embedding and retrieval stats (simulate or extract from state)
        query_embedding = state.get("query_embedding", np.zeros(384))
        chunk_embeddings = state.get("chunk_embeddings", [np.zeros(384)])
        chunk_similarities = state.get("chunk_similarities", [0.0])
        token_count = state.get("tokens", 0)
        latency = state.get("latency", 0)
        chunk_lengths = state.get("chunk_lengths", [0])
        user_feedback = state.get("user_feedback", 0.5)  # HITL: set by user or UI, defaults to 0.5 if not provided
        complexity_score = state.get("complexity_score", 0.0)  # Default to 0.0 if not provided
        # 3. State vector
        s = compute_state_vector(query_embedding, chunk_embeddings, token_count, latency, chunk_lengths, complexity_score, chunk_similarities)
        # 4. Q-Learning action
        action = self.q_agent.select_action(s)
        action_name = QLearningAgent.ACTIONS[action]
        # 5. Simulate reward (in real use, get from downstream)
        reward = calculate_reward(chunk_similarities[:5], latency, token_count, user_feedback)
        self.q_agent.update(s, reward)
        # 6. Suggestions
        suggestions = [f"Selected action: {action_name}"]
        if complexity_score > 0.5:
            suggestions.append("High mathematical complexity detected; consider hybrid or rerank strategy.")
        state["hallucination_suggestions"] = suggestions
        state["hallucination_completed"] = True
        self.metrics_history.append({"state": s.tolist(), "action": action_name, "reward": reward})
        return state

    def autonomous_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced autonomous processing with advanced RL capabilities."""
        print("[HallucinationAgent] Autonomous processing with RL optimization...")
        
        try:
            # Extract context for state computation
            query = state.get("request", "")
            retrieval_result = state.get("retrieval_result", {})
            
            if isinstance(retrieval_result, str):
                try:
                    retrieval_result = json.loads(retrieval_result)
                except:
                    retrieval_result = {"answer": retrieval_result, "sources": [], "confidence": 0.5}
            
            # Autonomous state assessment
            state_metrics = self._extract_state_metrics(query, retrieval_result, state)
            state_vector = self._compute_enhanced_state_vector(state_metrics)
            
            # Autonomous action selection with Q-learning
            autonomous_action = None
            if self.autonomous_mode:
                # Check if state manager has a recommendation
                autonomous_action = self.state_manager.get_best_action_for_state(state_vector)
            
            # Q-learning action selection (with autonomous override)
            if autonomous_action and np.random.rand() > self.q_agent.epsilon:
                action_idx = self.q_agent.ACTIONS.index(autonomous_action)
                print(f"  → Autonomous action selected: {autonomous_action}")
            else:
                action_idx = self.q_agent.select_action(state_vector)
            
            action_name = self.q_agent.ACTIONS[action_idx]
            
            # Execute action and calculate reward
            execution_result = self._execute_autonomous_action(action_name, state_metrics, state)
            reward = self.reward_calculator.calculate_reward(state_metrics, action_name)
            
            # Update learning systems
            self.q_agent.update(state_vector, reward)
            self.state_manager.update_performance(state_vector, action_name, reward)
            
            # Autonomous suggestions generation
            suggestions = self._generate_autonomous_suggestions(state_metrics, action_name, reward)
            
            # Update state with results
            state["hallucination_action"] = action_name
            state["hallucination_reward"] = reward
            state["hallucination_suggestions"] = suggestions
            state["hallucination_completed"] = True
            state["autonomous_metrics"] = {
                "action": action_name,
                "reward": reward,
                "state_similarity": float(np.linalg.norm(state_vector)),
                "learning_rate": self.q_agent.alpha,
                "exploration_rate": self.q_agent.epsilon
            }
            
            # Update metrics history
            self.metrics_history.append({
                "state": state_vector.tolist(), 
                "action": action_name, 
                "reward": reward,
                "timestamp": time.time(),
                "autonomous_decision": autonomous_action is not None
            })
            
            print(f"  ✅ Autonomous RL processing completed - Action: {action_name}, Reward: {reward:.3f}")
            return state
            
        except Exception as e:
            print(f"  ❌ Autonomous processing error: {e}")
            state["hallucination_error"] = str(e)
            state["hallucination_completed"] = True
            return state

    def _extract_state_metrics(self, query: str, retrieval_result: Dict, state: Dict) -> Dict[str, Any]:
        """Extract comprehensive metrics for autonomous state assessment."""
        metrics = {
            'query_length': len(query),
            'result_confidence': retrieval_result.get('confidence', 0.5),
            'sources_count': len(retrieval_result.get('sources', [])),
            'answer_length': len(str(retrieval_result.get('answer', ''))),
            'latency': state.get('processing_time', 0),
            'similarities': []
        }
        
        # Extract similarity scores if available
        if 'sources' in retrieval_result:
            for source in retrieval_result['sources']:
                if isinstance(source, dict) and 'similarity' in source:
                    metrics['similarities'].append(source['similarity'])
        
        return metrics

    def _execute_autonomous_action(self, action: str, metrics: Dict, state: Dict) -> Dict[str, Any]:
        """Execute autonomous action based on RL decision."""
        result = {'action_executed': action, 'effectiveness': 0.0}
        
        if action == 'vector_search':
            result['effectiveness'] = 0.8 if metrics.get('sources_count', 0) > 0 else 0.3
        elif action == 'rerank':
            result['effectiveness'] = 0.9 if len(metrics.get('similarities', [])) > 1 else 0.4
        elif action == 'expand':
            result['effectiveness'] = 0.7 if metrics.get('query_length', 0) < 100 else 0.5
        elif action == 'hybrid':
            result['effectiveness'] = 0.85
        elif action == 'cache':
            result['effectiveness'] = 0.95 if metrics.get('latency', 0) > 1000 else 0.6
        
        return result

    def _generate_autonomous_suggestions(self, metrics: Dict, action: str, reward: float) -> List[str]:
        """Generate autonomous improvement suggestions based on RL feedback."""
        suggestions = []
        
        if reward < 0.3:
            suggestions.append(f"Low reward ({reward:.3f}) for action '{action}' - consider parameter tuning")
        
        if metrics.get('latency', 0) > 3000:
            suggestions.append("High latency detected - recommend caching or optimization")
        
        if len(metrics.get('similarities', [])) == 0:
            suggestions.append("No similarity scores available - enhance vector search")
        
        avg_similarity = np.mean(metrics.get('similarities', [0.5]))
        if avg_similarity < 0.3:
            suggestions.append(f"Low similarity scores ({avg_similarity:.3f}) - improve retrieval strategy")
        
        if action == 'hybrid' and reward > 0.8:
            suggestions.append("Hybrid approach working well - continue similar strategy")
        
        return suggestions

    def get_metrics_report(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
        latest = self.metrics_history[-1]
        avg_reward = float(np.mean([m["reward"] for m in self.metrics_history]))
        return {
            "latest": latest,
            "average_reward": avg_reward,
            "total_requests": len(self.metrics_history)
        }
