"""
ITRS (Iterative Transparent Reasoning System) Component Template

Reusable AI-OS component implementing the ITRS reasoning methodology 
for zero-heuristic decision making with iterative refinement.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base.component import AIServiceComponent
from ..config.models import ReasoningConfig, ComponentType, create_config


logger = logging.getLogger(__name__)


class RefinementStrategy(str, Enum):
    """Available refinement strategies for ITRS"""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    CONSENSUS_BUILDING = "consensus_building"
    PERSPECTIVE_SWITCHING = "perspective_switching"
    EVIDENCE_INTEGRATION = "evidence_integration"
    CONTRARIAN_ANALYSIS = "contrarian_analysis"


class SessionStatus(str, Enum):
    """Status of reasoning sessions"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StoppingCriterion(str, Enum):
    """Criteria for stopping reasoning sessions"""
    QUALITY_THRESHOLD = "quality_threshold"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    USER_INTERVENTION = "user_intervention"
    CONVERGENCE = "convergence"


@dataclass
class ThoughtNode:
    """Individual thought/reasoning step"""
    content: str
    confidence: float
    reasoning_path: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningSession:
    """A complete reasoning session with iterative refinement"""
    id: str
    query: str
    initial_response: str
    current_iteration: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    thoughts: List[ThoughtNode] = field(default_factory=list)
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    strategies_used: List[RefinementStrategy] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    stopping_reason: Optional[StoppingCriterion] = None


class ITRSReasoningComponent(AIServiceComponent):
    """
    ITRS Reasoning Component - Implements Iterative Transparent Reasoning System
    
    Features:
    - Zero-heuristic decision making
    - Six refinement strategies for iterative improvement
    - Quality tracking and convergence detection
    - Session management with pause/resume capabilities
    - Transparent reasoning with detailed thought tracking
    - Configurable stopping criteria and quality thresholds
    """
    
    def __init__(self, config: Union[ReasoningConfig, str, Dict[str, Any]]):
        """Initialize ITRS Reasoning Component"""
        
        # Ensure we have the right type
        if isinstance(config, dict):
            config['type'] = ComponentType.REASONING
            config = create_config(ComponentType.REASONING, **config)
        elif isinstance(config, str):
            super().__init__(config)
            if self.config.type != ComponentType.REASONING:
                raise ValueError(f"Configuration type must be 'reasoning', got '{self.config.type}'")
            return
        
        super().__init__(config)
        
        # ITRS-specific configuration
        self.max_iterations = getattr(self.config, 'max_iterations', 10)
        self.quality_threshold = getattr(self.config, 'quality_threshold', 0.85)
        self.time_limit = getattr(self.config, 'time_limit', 300)  # seconds
        self.convergence_threshold = getattr(self.config, 'convergence_threshold', 0.05)
        self.min_confidence = getattr(self.config, 'min_confidence', 0.7)
        
        # Session management
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.completed_sessions: Dict[str, ReasoningSession] = {}
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness: Dict[RefinementStrategy, List[float]] = {
            strategy: [] for strategy in RefinementStrategy
        }
        
        # Component metrics
        self.total_sessions = 0
        self.total_iterations = 0
        self.total_processing_time = 0.0
    
    def _initialize(self):
        """Initialize ITRS component"""
        self.logger.info(f"Initializing ITRS Reasoning Component: {self.name}")
        self.logger.info(f"Configuration: max_iterations={self.max_iterations}, "
                        f"quality_threshold={self.quality_threshold}, "
                        f"time_limit={self.time_limit}s")
    
    def _execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute ITRS reasoning for a query"""
        return asyncio.run(self._execute_async(query, **kwargs))
    
    async def _execute_async(self, query: str, **kwargs) -> Dict[str, Any]:
        """Async execution of ITRS reasoning"""
        session_id = await self.start_reasoning_session(query, **kwargs)
        
        # Run complete session
        session_result = await self.run_complete_session(session_id)
        
        return {
            "session_id": session_id,
            "final_response": session_result.get("best_response", ""),
            "reasoning_path": session_result.get("reasoning_path", []),
            "quality_score": session_result.get("final_quality", 0.0),
            "iterations": session_result.get("iterations", 0),
            "strategies_used": session_result.get("strategies_used", []),
            "session_summary": session_result
        }
    
    async def start_reasoning_session(
        self,
        query: str,
        initial_response: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Start a new reasoning session"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Generate initial response if not provided
        if not initial_response:
            initial_response = await self._generate_initial_response(query)
        
        # Create session
        session = ReasoningSession(
            id=session_id,
            query=query,
            initial_response=initial_response,
            thoughts=[ThoughtNode(
                content=initial_response,
                confidence=0.5,  # Initial uncertainty
                reasoning_path=["initial_response"]
            )]
        )
        
        # Add initial quality score
        initial_quality = await self._evaluate_quality(initial_response, query)
        session.quality_scores.append(initial_quality)
        
        # Store session
        self.active_sessions[session_id] = session
        self.total_sessions += 1
        
        self.logger.info(f"Started reasoning session {session_id} with initial quality {initial_quality:.3f}")
        return session_id
    
    async def iterate_session(
        self,
        session_id: str,
        user_feedback: Optional[str] = None,
        preferred_strategy: Optional[RefinementStrategy] = None
    ) -> Dict[str, Any]:
        """Perform one iteration of reasoning refinement"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is not active")
        
        # Check stopping criteria
        should_stop, stopping_reason = self._should_stop_session(session)
        if should_stop:
            await self._complete_session(session, stopping_reason)
            return {
                "session_completed": True,
                "stopping_reason": stopping_reason.value,
                "final_quality": session.quality_scores[-1] if session.quality_scores else 0.0
            }
        
        # Select refinement strategy
        strategy = preferred_strategy or self._select_strategy(session)
        
        # Generate refinement
        current_thought = session.thoughts[-1]
        refined_thought = await self._apply_strategy(
            strategy, current_thought, session, user_feedback
        )
        
        # Evaluate quality
        new_quality = await self._evaluate_quality(refined_thought.content, session.query)
        
        # Update session
        session.thoughts.append(refined_thought)
        session.quality_scores.append(new_quality)
        session.strategies_used.append(strategy)
        session.current_iteration += 1
        
        # Track strategy effectiveness
        if len(session.quality_scores) >= 2:
            improvement = new_quality - session.quality_scores[-2]
            self.strategy_effectiveness[strategy].append(max(0, improvement))
        
        # Add to refinement history
        session.refinement_history.append({
            "iteration": session.current_iteration,
            "strategy": strategy.value,
            "quality_before": session.quality_scores[-2] if len(session.quality_scores) >= 2 else 0,
            "quality_after": new_quality,
            "improvement": new_quality - (session.quality_scores[-2] if len(session.quality_scores) >= 2 else 0),
            "confidence": refined_thought.confidence,
            "user_feedback": user_feedback
        })
        
        self.total_iterations += 1
        
        self.logger.info(f"Iteration {session.current_iteration} for session {session_id}: "
                        f"strategy={strategy.value}, quality={new_quality:.3f}")
        
        return {
            "session_completed": False,
            "iteration": session.current_iteration,
            "strategy_used": strategy.value,
            "quality_score": new_quality,
            "confidence": refined_thought.confidence,
            "can_continue": not self._should_stop_session(session)[0]
        }
    
    async def run_complete_session(self, session_id: str) -> Dict[str, Any]:
        """Run a complete reasoning session until stopping criteria are met"""
        
        while session_id in self.active_sessions:
            try:
                result = await self.iterate_session(session_id)
                if result.get("session_completed", False):
                    break
                    
                # Safety check
                session = self.active_sessions.get(session_id)
                if session and session.current_iteration >= self.max_iterations:
                    await self._complete_session(session, StoppingCriterion.MAX_ITERATIONS)
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in session {session_id}: {e}")
                session = self.active_sessions.get(session_id)
                if session:
                    session.status = SessionStatus.FAILED
                    self._move_to_completed(session)
                break
        
        # Return session summary
        if session_id in self.completed_sessions:
            return self._get_session_summary(self.completed_sessions[session_id])
        else:
            return {"error": "Session not completed properly"}
    
    async def _generate_initial_response(self, query: str) -> str:
        """Generate initial response to query"""
        # This would integrate with the actual LLM
        # For now, return a structured initial response
        return f"Initial analysis of: {query}\n\nThis requires careful consideration of multiple factors and perspectives."
    
    def _select_strategy(self, session: ReasoningSession) -> RefinementStrategy:
        """Select the best refinement strategy based on session context"""
        
        # Strategy selection logic based on iteration and effectiveness
        current_iteration = session.current_iteration
        
        # Early iterations: explore different perspectives
        if current_iteration == 0:
            return RefinementStrategy.DEPTH_FIRST
        elif current_iteration == 1:
            return RefinementStrategy.PERSPECTIVE_SWITCHING
        elif current_iteration == 2:
            return RefinementStrategy.EVIDENCE_INTEGRATION
        
        # Later iterations: use most effective strategies
        else:
            # Calculate average effectiveness for each strategy
            strategy_scores = {}
            for strategy, improvements in self.strategy_effectiveness.items():
                if improvements:
                    strategy_scores[strategy] = sum(improvements) / len(improvements)
                else:
                    strategy_scores[strategy] = 0.0
            
            # Select strategy with highest average improvement
            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
                return best_strategy
            else:
                # Fallback to consensus building
                return RefinementStrategy.CONSENSUS_BUILDING
    
    async def _apply_strategy(
        self,
        strategy: RefinementStrategy,
        current_thought: ThoughtNode,
        session: ReasoningSession,
        user_feedback: Optional[str] = None
    ) -> ThoughtNode:
        """Apply a specific refinement strategy"""
        
        # Strategy-specific refinement logic
        if strategy == RefinementStrategy.DEPTH_FIRST:
            return await self._depth_first_refinement(current_thought, session)
        elif strategy == RefinementStrategy.BREADTH_FIRST:
            return await self._breadth_first_refinement(current_thought, session)
        elif strategy == RefinementStrategy.PERSPECTIVE_SWITCHING:
            return await self._perspective_switching_refinement(current_thought, session)
        elif strategy == RefinementStrategy.EVIDENCE_INTEGRATION:
            return await self._evidence_integration_refinement(current_thought, session)
        elif strategy == RefinementStrategy.CONSENSUS_BUILDING:
            return await self._consensus_building_refinement(current_thought, session)
        elif strategy == RefinementStrategy.CONTRARIAN_ANALYSIS:
            return await self._contrarian_analysis_refinement(current_thought, session)
        else:
            # Fallback to depth-first
            return await self._depth_first_refinement(current_thought, session)
    
    async def _depth_first_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Deep dive into specific aspects of the current reasoning"""
        refined_content = f"Deep analysis building on: {thought.content}\n\n" \
                         f"Examining underlying assumptions and logical foundations..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=min(thought.confidence + 0.1, 1.0),
            reasoning_path=thought.reasoning_path + ["depth_first_analysis"],
            evidence=thought.evidence + ["detailed_examination"],
            assumptions=thought.assumptions + ["deeper_logical_structure"]
        )
    
    async def _breadth_first_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Explore alternative approaches and broader context"""
        refined_content = f"Broader perspective on: {thought.content}\n\n" \
                         f"Considering alternative approaches and wider implications..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=thought.confidence + 0.05,
            reasoning_path=thought.reasoning_path + ["breadth_first_exploration"],
            evidence=thought.evidence + ["alternative_perspectives"],
            assumptions=thought.assumptions + ["broader_context"]
        )
    
    async def _perspective_switching_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Switch to different stakeholder or analytical perspectives"""
        refined_content = f"Alternative perspective on: {thought.content}\n\n" \
                         f"Viewing from different stakeholder and analytical angles..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=thought.confidence + 0.08,
            reasoning_path=thought.reasoning_path + ["perspective_switch"],
            evidence=thought.evidence + ["multi_stakeholder_view"],
            assumptions=thought.assumptions + ["different_viewpoints"]
        )
    
    async def _evidence_integration_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Integrate additional evidence and strengthen reasoning"""
        refined_content = f"Evidence-strengthened analysis: {thought.content}\n\n" \
                         f"Incorporating additional evidence and data points..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=min(thought.confidence + 0.15, 1.0),
            reasoning_path=thought.reasoning_path + ["evidence_integration"],
            evidence=thought.evidence + ["additional_data", "supporting_research"],
            assumptions=thought.assumptions
        )
    
    async def _consensus_building_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Build consensus from multiple reasoning paths"""
        refined_content = f"Consensus view incorporating: {thought.content}\n\n" \
                         f"Synthesizing multiple reasoning approaches..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=min(thought.confidence + 0.12, 1.0),
            reasoning_path=thought.reasoning_path + ["consensus_building"],
            evidence=thought.evidence + ["synthesized_views"],
            assumptions=thought.assumptions + ["consensus_assumptions"]
        )
    
    async def _contrarian_analysis_refinement(self, thought: ThoughtNode, session: ReasoningSession) -> ThoughtNode:
        """Challenge current reasoning with contrarian analysis"""
        refined_content = f"Contrarian analysis of: {thought.content}\n\n" \
                         f"Challenging assumptions and exploring counter-arguments..."
        
        return ThoughtNode(
            content=refined_content,
            confidence=thought.confidence + 0.06,
            reasoning_path=thought.reasoning_path + ["contrarian_analysis"],
            evidence=thought.evidence + ["counter_evidence"],
            assumptions=thought.assumptions + ["challenged_assumptions"]
        )
    
    async def _evaluate_quality(self, content: str, query: str) -> float:
        """Evaluate the quality of reasoning content"""
        # Quality evaluation based on multiple criteria
        # This would integrate with actual quality assessment
        
        # Simple quality heuristics for now
        quality_score = 0.0
        
        # Length and detail
        if len(content) > 100:
            quality_score += 0.2
        if len(content) > 300:
            quality_score += 0.1
        
        # Structured reasoning indicators
        if "analysis" in content.lower():
            quality_score += 0.1
        if "evidence" in content.lower():
            quality_score += 0.1
        if "perspective" in content.lower():
            quality_score += 0.1
        
        # Connection to original query
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        quality_score += overlap * 0.3
        
        # Ensure score is between 0 and 1
        return min(max(quality_score, 0.0), 1.0)
    
    def _should_stop_session(self, session: ReasoningSession) -> tuple[bool, Optional[StoppingCriterion]]:
        """Check if session should stop based on stopping criteria"""
        
        # Max iterations
        if session.current_iteration >= self.max_iterations:
            return True, StoppingCriterion.MAX_ITERATIONS
        
        # Quality threshold
        if session.quality_scores and session.quality_scores[-1] >= self.quality_threshold:
            return True, StoppingCriterion.QUALITY_THRESHOLD
        
        # Time limit
        elapsed_time = (datetime.now() - session.created_at).total_seconds()
        if elapsed_time >= self.time_limit:
            return True, StoppingCriterion.TIME_LIMIT
        
        # Convergence (quality not improving)
        if len(session.quality_scores) >= 3:
            recent_scores = session.quality_scores[-3:]
            if max(recent_scores) - min(recent_scores) < self.convergence_threshold:
                return True, StoppingCriterion.CONVERGENCE
        
        return False, None
    
    async def _complete_session(self, session: ReasoningSession, reason: StoppingCriterion):
        """Complete a reasoning session"""
        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.now()
        session.stopping_reason = reason
        
        self._move_to_completed(session)
        
        self.logger.info(f"Completed session {session.id}: reason={reason.value}, "
                        f"iterations={session.current_iteration}, "
                        f"final_quality={session.quality_scores[-1] if session.quality_scores else 0:.3f}")
    
    def _move_to_completed(self, session: ReasoningSession):
        """Move session from active to completed"""
        if session.id in self.active_sessions:
            del self.active_sessions[session.id]
        
        self.completed_sessions[session.id] = session
        
        # Keep only recent sessions
        if len(self.completed_sessions) > 50:
            oldest_sessions = sorted(
                self.completed_sessions.items(),
                key=lambda x: x[1].created_at
            )[:-50]
            
            for old_id, _ in oldest_sessions:
                del self.completed_sessions[old_id]
    
    def _get_session_summary(self, session: ReasoningSession) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        best_quality_idx = session.quality_scores.index(max(session.quality_scores)) if session.quality_scores else 0
        best_thought = session.thoughts[best_quality_idx] if session.thoughts else None
        
        return {
            "session_id": session.id,
            "query": session.query,
            "status": session.status.value,
            "iterations": session.current_iteration,
            "best_response": best_thought.content if best_thought else session.initial_response,
            "final_quality": session.quality_scores[-1] if session.quality_scores else 0.0,
            "best_quality": max(session.quality_scores) if session.quality_scores else 0.0,
            "quality_progression": session.quality_scores,
            "strategies_used": [s.value for s in session.strategies_used],
            "reasoning_path": best_thought.reasoning_path if best_thought else [],
            "evidence": best_thought.evidence if best_thought else [],
            "assumptions": best_thought.assumptions if best_thought else [],
            "stopping_reason": session.stopping_reason.value if session.stopping_reason else None,
            "created_at": session.created_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "refinement_history": session.refinement_history
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a reasoning session"""
        session = None
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        elif session_id in self.completed_sessions:
            session = self.completed_sessions[session_id]
        else:
            raise ValueError(f"Session {session_id} not found")
        
        return self._get_session_summary(session)
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get component usage statistics"""
        strategy_effectiveness = {}
        for strategy, improvements in self.strategy_effectiveness.items():
            if improvements:
                strategy_effectiveness[strategy.value] = {
                    "average_improvement": sum(improvements) / len(improvements),
                    "total_uses": len(improvements),
                    "max_improvement": max(improvements)
                }
            else:
                strategy_effectiveness[strategy.value] = {
                    "average_improvement": 0.0,
                    "total_uses": 0,
                    "max_improvement": 0.0
                }
        
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "total_iterations": self.total_iterations,
            "strategy_effectiveness": strategy_effectiveness,
            "configuration": {
                "max_iterations": self.max_iterations,
                "quality_threshold": self.quality_threshold,
                "time_limit": self.time_limit,
                "convergence_threshold": self.convergence_threshold
            }
        }
    
    def _health_check(self) -> Dict[str, Any]:
        """ITRS component health check"""
        return {
            "status": "ok",
            "component_type": "reasoning",
            "reasoning_method": "ITRS",
            "active_sessions": len(self.active_sessions),
            "total_sessions": self.total_sessions,
            "configuration_valid": True
        }