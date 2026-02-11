"""
LLM modules for DataWarrior-AI

Architecture:
- LARGE MODELS (GPT-4/Claude): Two-Stage Workflow
  Stage 1: Planning with short descriptions
  Stage 2: Parameterization with detailed schemas

- SMALL MODELS (Ollama): Semantic Search + Single-Stage
  Preprocessing: top-K via embeddings
  Single call: selection + parameterization
"""

from .llm_client import (
    LLMClient,
    LLMConfig,
    ModelProvider,
    create_openai_client,
    create_anthropic_client,
    create_ollama_client
)

from .semantic_search import (
    SemanticSearch,
    get_relevant_tools
)

from .orchestrator import (
    DataWarriorOrchestrator,
    ToolCall,
    OrchestratorResult,
    create_orchestrator_openai,
    create_orchestrator_anthropic,
    create_orchestrator_ollama,
    MACRO_DESCRIPTIONS,
    MACRO_SCHEMAS
)

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMConfig", 
    "ModelProvider",
    "create_openai_client",
    "create_anthropic_client",
    "create_ollama_client",
    
    # Semantic Search
    "SemanticSearch",
    "get_relevant_tools",
    
    # Orchestrator
    "DataWarriorOrchestrator",
    "ToolCall",
    "OrchestratorResult",
    "create_orchestrator_openai",
    "create_orchestrator_anthropic",
    "create_orchestrator_ollama",
    
    # Data
    "MACRO_DESCRIPTIONS",
    "MACRO_SCHEMAS"
]
