#!/usr/bin/env python3
"""
LLM Orchestrator for DataWarrior-Agent v1

Architecture:
- LARGE MODELS (GPT-4/Claude): Two-Stage Workflow + Dataset Access
- SMALL MODELS (Ollama): Semantic Search + Single-Stage + Dataset Access

Fixes v6.2:
- Improved Single-Stage prompt to avoid inventing parameters
- Unified Stage 3 for all models
- Better handling of malformed JSON responses
"""

import json
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .llm_client import (
    LLMClient, 
    LLMConfig,
    ModelProvider, 
    create_openai_client, 
    create_anthropic_client, 
    create_ollama_client
)
from .semantic_search import SemanticSearch
from .token_manager import (
    TokenLimits,
    LARGE_MODEL_LIMITS,
    SMALL_MODEL_LIMITS,
    estimate_tokens,
    estimate_messages_tokens,
    get_limits_for_model,
    compress_history
)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class ToolCall:
    """Represents a tool call"""
    name: str
    arguments: Dict[str, Any]
    reason: str = ""


@dataclass 
class OrchestratorResult:
    """Orchestration result"""
    success: bool
    tool_calls: List[ToolCall]
    response: str
    reasoning: str = ""
    error: Optional[str] = None
    stage: str = ""
    warning: Optional[str] = None


# ============================================
# CONFIGURATION
# ============================================

CONFIG_DIR = Path(__file__).parent.parent / "config"
MACRO_DESCRIPTIONS_PATH = CONFIG_DIR / "macro_descriptions.json"
MACRO_SCHEMAS_PATH = CONFIG_DIR / "macro_schemas.json"


def load_macro_descriptions() -> Dict[str, str]:
    """Loads short descriptions for Stage 1"""
    if MACRO_DESCRIPTIONS_PATH.exists():
        with open(MACRO_DESCRIPTIONS_PATH, 'r') as f:
            data = json.load(f)
            return data.get("macros", {})
    return {}


def load_macro_schemas() -> Dict[str, dict]:
    """Loads detailed schemas for Stage 2"""
    if MACRO_SCHEMAS_PATH.exists():
        with open(MACRO_SCHEMAS_PATH, 'r') as f:
            data = json.load(f)
            return data.get("schemas", {})
    return {}


# Load data at startup
MACRO_DESCRIPTIONS = load_macro_descriptions()
MACRO_SCHEMAS = load_macro_schemas()


# ============================================
# ORCHESTRATOR
# ============================================

class DataWarriorOrchestrator:
    """
    Orchestrates LLM calls to control DataWarrior
    
    Features:
    - Two-Stage (large models) / Single-Stage (small models)
    - Token management with automatic compression
    - Dataset access to answer questions about the data
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.is_large_model = LLMClient.is_large_model(llm_client.config.model)
        self.limits = get_limits_for_model(self.is_large_model)
        self.semantic_search = SemanticSearch() if not self.is_large_model else None
        
        # Client for compression (always GPT-4o-mini)
        self.compression_llm: Optional[LLMClient] = None
        
        # Histories
        self.planning_history: List[Dict[str, str]] = []
        self.execution_history: List[Dict[str, str]] = []
        self.conversation_history: List[Dict[str, str]] = []
        
        # Dataset context
        self.current_dataset_context: Optional[str] = None
        
        print(f"[Orchestrator] Model: {llm_client.config.model}")
        print(f"[Orchestrator] Strategy: {'Two-Stage' if self.is_large_model else 'Semantic Search + Single-Stage'}")
        print(f"[Orchestrator] Token limits - CSV: {self.limits.csv_warning_threshold}, History: {self.limits.history_max_tokens}")
    
    def set_compression_llm(self, api_key: str):
        """Configures the LLM for compression (GPT-4o-mini recommended)"""
        self.compression_llm = create_openai_client(api_key, "gpt-4o-mini")
        print("[Orchestrator] Compression LLM configured: gpt-4o-mini")
    
    # ============================================
    # DATASET CONTEXT MANAGEMENT
    # ============================================
    
    def set_dataset_context(self, dataset_content: str, metadata: Dict):
        """
        Sets the dataset context for questions about the data.
        Called after read_dataset.
        """
        tokens = estimate_tokens(dataset_content)
        
        warning = None
        if tokens > self.limits.csv_warning_threshold:
            warning = f"⚠️ Large dataset ({tokens} tokens). Analysis may be incomplete."
        
        # Format the context
        self.current_dataset_context = f"""[CURRENT DATASET]
Rows: {metadata.get('rows', 'unknown')}
Columns: {', '.join(metadata.get('columns', []))}

[DATA]
{dataset_content}
[END DATASET]"""
        
        print(f"[Orchestrator] Dataset context set: {tokens} tokens")
        return warning
    
    def clear_dataset_context(self):
        """Clears the dataset context"""
        self.current_dataset_context = None
    
    # ============================================
    # MAIN ENTRY POINT
    # ============================================
    
    async def process(self, user_message: str) -> OrchestratorResult:
        """Processes a user request"""
        
        # Check and compress history if needed
        compression_warning = await self._check_and_compress_history()
        
        # Process based on model type
        if self.is_large_model:
            result = await self._process_two_stage(user_message)
        else:
            result = await self._process_single_stage(user_message)
        
        # Add compression warning if present
        if compression_warning and not result.warning:
            result.warning = compression_warning
        elif compression_warning:
            result.warning = f"{compression_warning}\n{result.warning}"
        
        return result
    
    async def _check_and_compress_history(self) -> Optional[str]:
        """Checks and compresses history if needed"""
        
        if self.is_large_model:
            history = self.planning_history
        else:
            history = self.conversation_history
        
        tokens = estimate_messages_tokens(history)
        
        if tokens <= self.limits.history_max_tokens:
            return None
        
        print(f"[Orchestrator] History compression needed: {tokens} > {self.limits.history_max_tokens}")
        
        # Use the compression LLM or the main LLM
        llm_for_compression = self.compression_llm or self.llm
        
        new_history, compression_msg = await compress_history(
            history,
            self.limits,
            llm_for_compression
        )
        
        # Update history
        if self.is_large_model:
            self.planning_history = new_history
        else:
            self.conversation_history = new_history
        
        return compression_msg
    
    # ============================================
    # TWO-STAGE WORKFLOW (Large models)
    # ============================================
    
    async def _process_two_stage(self, user_message: str) -> OrchestratorResult:
        """Two-Stage Workflow for GPT-4/Claude"""
        
        print(f"\n[Two-Stage] Processing: {user_message[:50]}...")
        
        # ===== STAGE 1: PLANNING + RESPONSE =====
        print("[Stage 1] Planning & Response Generation...")
        
        stage1_result = await self._stage1_planning(user_message)
        
        if not stage1_result["success"]:
            return OrchestratorResult(
                success=False,
                tool_calls=[],
                response=stage1_result.get("response", "An error occurred."),
                error=stage1_result.get("error"),
                stage="planning"
            )
        
        selected_macros = stage1_result["selected_macros"]
        plan = stage1_result["plan"]
        user_response = stage1_result["response"]
        warning = stage1_result.get("warning")
        
        print(f"[Stage 1] Selected macros: {selected_macros}")
        print(f"[Stage 1] Response: {user_response[:100]}...")
        
        # If no macro selected → conversational response
        if not selected_macros:
            return OrchestratorResult(
                success=True,
                tool_calls=[],
                response=user_response,
                reasoning=plan,
                stage="planning",
                warning=warning
            )
        
        # ===== STAGE 2: PARAMETERIZATION =====
        print(f"[Stage 2] Parameterization - {len(selected_macros)} macros...")
        
        stage2_result = await self._stage2_parameterization(
            user_message, 
            selected_macros, 
            plan
        )
        
        # Keep the Stage 1 response
        if stage2_result.success and stage2_result.tool_calls:
            stage2_result.response = user_response
        
        stage2_result.warning = warning
        
        return stage2_result
    
    async def _stage1_planning(self, user_message: str) -> dict:
        """Stage 1: Macro selection + Response generation"""
        
        system_prompt = self._get_stage1_prompt()
        
        # Add to history
        self.planning_history.append({"role": "user", "content": user_message})
        
        try:
            response = await self.llm.chat(
                messages=self.planning_history,
                system_prompt=system_prompt
            )
            
            result = self._parse_stage1_response(response)
            
            self.planning_history.append({"role": "assistant", "content": response})
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selected_macros": [],
                "plan": "",
                "response": f"Sorry, an error occurred: {str(e)}"
            }
    
    async def _stage2_parameterization(
        self, 
        user_message: str, 
        selected_macros: List[str],
        plan: str
    ) -> OrchestratorResult:
        """Stage 2: Parameterization of selected macros"""
        
        system_prompt = self._get_stage2_prompt(selected_macros)
        
        enriched_message = f"""User request: {user_message}

Execution plan from planning stage:
{plan}

Now parameterize the selected tools with correct arguments."""
        
        self.execution_history = [{"role": "user", "content": enriched_message}]
        
        try:
            response = await self.llm.chat(
                messages=self.execution_history,
                system_prompt=system_prompt
            )
            
            result = self._parse_stage2_response(response)
            result.stage = "execution"
            
            self.execution_history.append({"role": "assistant", "content": response})
            
            return result
            
        except Exception as e:
            return OrchestratorResult(
                success=False,
                tool_calls=[],
                response=f"Error during parameterization: {str(e)}",
                error=str(e),
                stage="execution"
            )
    
    def _get_stage1_prompt(self) -> str:
        """Prompt for Stage 1 (Planning + Response Generation)"""
        
        macro_list = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in MACRO_DESCRIPTIONS.items()
        ])
        
        # Add dataset context if available
        dataset_section = ""
        if self.current_dataset_context:
            dataset_section = f"""

=== CURRENT DATASET CONTEXT ===
{self.current_dataset_context}
=== END DATASET CONTEXT ===

You have access to the dataset above. Use this information to answer questions about the data.
"""
        
        return f"""You are a helpful assistant for DataWarrior, a cheminformatics platform.
You help users analyze molecular data by selecting appropriate tools AND providing helpful responses.

AVAILABLE TOOLS:
{macro_list}
{dataset_section}

=== RESPONSE FORMAT (JSON only) ===
{{
    "selected_macros": ["macro1", "macro2"],
    "plan": "Technical execution plan (for internal use)",
    "response": "REQUIRED: Natural language response to show the user"
}}

=== CRITICAL RULES ===

1. The "response" field is MANDATORY and must NEVER be empty
2. ALWAYS check the conversation history to understand context
3. For ANY user input, you MUST provide a helpful response

=== HANDLING DATA QUESTIONS ===

When the user asks about the dataset content (columns, values, statistics, etc.):

**If dataset context is available above:**
- Answer directly using the data provided
- You can calculate statistics, count values, filter data mentally
- Provide specific numbers and values from the data

**If NO dataset context is available:**
- Select "read_dataset" macro to load the data first
- response: "I will first load the data to be able to answer your question."

=== HOW TO HANDLE DIFFERENT INPUTS ===

**ACTION REQUESTS** (calculate, open, filter, save...):
- selected_macros: [relevant tools]
- response: "I will [action]..."

**DATA QUESTIONS** (columns, values, statistics, "how many...", "which columns..."):
- If dataset context available: answer directly, selected_macros: []
- If no context: selected_macros: ["read_dataset"]

**CONVERSATIONAL** (what, why, how, can you...):
- selected_macros: []
- response: Full answer

**CONFIRMATIONS** (yes, no):
- Check history for context
- Act accordingly

=== EXAMPLES ===

User: "Calculate logP"
{{"selected_macros": ["calculate_descriptors"], "plan": "Calculate logP", "response": "I will calculate the logP for your molecules."}}

User: "What columns are available?"
(No dataset context)
{{"selected_macros": ["read_dataset"], "plan": "Load dataset to see columns", "response": "I will load the data to see the available columns."}}

User: "What columns are available?"
(Dataset context shows: Structure, logP, MW, TPSA)
{{"selected_macros": [], "plan": "", "response": "The available columns are: Structure, logP, MW, TPSA."}}

User: "How many molecules have a logP > 3?"
(Dataset context available with logP values)
{{"selected_macros": [], "plan": "", "response": "According to the data, 15 molecules have a logP greater than 3."}}

=== REMEMBER ===
- NEVER return an empty response
- Use the dataset context when available to answer data questions directly
- If you need data but don't have it, use read_dataset
- Respond in the same language as the user
"""
    
    def _get_stage2_prompt(self, selected_macros: List[str]) -> str:
        """Prompt for Stage 2 (Parameterization)"""
        
        schemas_text = ""
        for macro_name in selected_macros:
            if macro_name in MACRO_SCHEMAS:
                schema = MACRO_SCHEMAS[macro_name]
                schemas_text += f"\n### {macro_name}\n"
                schemas_text += f"Description: {schema.get('description', '')}\n"
                schemas_text += f"Parameters: {json.dumps(schema.get('parameters', {}), indent=2)}\n"
                
                if "valid_descriptors" in schema:
                    schemas_text += f"Valid descriptors: {json.dumps(schema['valid_descriptors'], indent=2)}\n"
                if "aliases" in schema:
                    schemas_text += f"Aliases: {json.dumps(schema['aliases'])}\n"
        
        return f"""You are a parameterization assistant for DataWarrior.

YOUR TASK: Generate the correct parameters for each selected tool.

SELECTED TOOLS AND THEIR SCHEMAS:
{schemas_text}

RESPONSE FORMAT (JSON only):
{{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{"param1": "value1"}},
            "reason": "Brief explanation"
        }}
    ]
}}

CRITICAL RULES:
1. Use EXACT parameter names from schemas
2. Use EXACT descriptor names
3. If no parameters required, use empty object {{}}
"""
    
    def _parse_stage1_response(self, response: str) -> dict:
        """Parses the JSON response from Stage 1"""
        try:
            clean = self._clean_json_response(response)
            data = json.loads(clean)
            
            user_response = data.get("response", "").strip()
            
            # Fallbacks
            if not user_response:
                if data.get("plan"):
                    user_response = data["plan"]
                elif data.get("selected_macros"):
                    macros = ", ".join(data["selected_macros"])
                    user_response = f"I will execute: {macros}"
                else:
                    user_response = "I am here to help you. What would you like to do?"
            
            return {
                "success": True,
                "selected_macros": data.get("selected_macros", []),
                "plan": data.get("plan", ""),
                "response": user_response
            }
            
        except json.JSONDecodeError as e:
            clean_response = response.strip()
            
            if not clean_response.startswith("{"):
                return {
                    "success": True,
                    "selected_macros": [],
                    "plan": "",
                    "response": clean_response
                }
            
            return {
                "success": True,
                "selected_macros": [],
                "plan": "",
                "response": "I didn't understand well. Could you rephrase?",
                "error": f"JSON parse error: {e}"
            }
    
    def _parse_stage2_response(self, response: str) -> OrchestratorResult:
        """Parses the JSON response from Stage 2"""
        try:
            clean = self._clean_json_response(response)
            data = json.loads(clean)
            
            tool_calls = []
            for tc in data.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    reason=tc.get("reason", "")
                ))
            
            return OrchestratorResult(
                success=True,
                tool_calls=tool_calls,
                response=data.get("response", ""),
                reasoning=""
            )
            
        except json.JSONDecodeError as e:
            return OrchestratorResult(
                success=True,
                tool_calls=[],
                response="Error processing the request.",
                error=f"JSON parse error: {e}"
            )
    
    # ============================================
    # SINGLE-STAGE WORKFLOW (Small models)
    # ============================================
    
    async def _process_single_stage(self, user_message: str) -> OrchestratorResult:
        """Single-Stage for Ollama/small models"""
        
        print(f"\n[Single-Stage] Processing: {user_message[:50]}...")
        
        # Semantic search to find relevant tools
        results = self.semantic_search.search(
            user_message, 
            MACRO_DESCRIPTIONS, 
            top_k=5
        )
        
        relevant_macros = [name for name, score in results if score > 0.3]
        if len(relevant_macros) < 3:
            relevant_macros = [name for name, _ in results[:5]]
        
        # Always include read_dataset for data questions
        if "read_dataset" not in relevant_macros:
            relevant_macros.append("read_dataset")
        
        print(f"[Semantic Search] Retrieved: {relevant_macros}")
        
        system_prompt = self._get_single_stage_prompt(relevant_macros)
        
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            response = await self.llm.chat(
                messages=self.conversation_history,
                system_prompt=system_prompt
            )
            
            result = self._parse_single_stage_response(response)
            result.stage = "single-stage"
            
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return result
            
        except Exception as e:
            return OrchestratorResult(
                success=False,
                tool_calls=[],
                response=f"Error: {str(e)}",
                error=str(e),
                stage="single-stage"
            )
    
    def _get_single_stage_prompt(self, relevant_macros: List[str]) -> str:
        """Optimized prompt for small models (Ollama) - v6.2"""
        
        tools_text = ""
        for macro_name in relevant_macros:
            desc = MACRO_DESCRIPTIONS.get(macro_name, "")
            schema = MACRO_SCHEMAS.get(macro_name, {})
            params = schema.get('parameters', {}).get('properties', {})
            required = schema.get('parameters', {}).get('required', [])
            
            tools_text += f"\n### {macro_name}\n"
            tools_text += f"Description: {desc}\n"
            
            if params:
                param_names = list(params.keys())
                tools_text += f"Parameters: {param_names}\n"
                tools_text += f"Required: {required}\n"
            else:
                tools_text += "Parameters: NONE - use empty arguments {}\n"
        
        # Dataset context
        dataset_section = ""
        has_dataset = False
        if self.current_dataset_context:
            has_dataset = True
            context = self.current_dataset_context
            # Limit for small models
            max_chars = self.limits.csv_warning_threshold * 4
            if len(context) > max_chars:
                context = context[:max_chars] + "\n[TRUNCATED...]"
            dataset_section = f"""

=== CURRENT DATASET (USE THIS DATA TO ANSWER) ===
{context}
=== END DATASET ==="""
        
        # Clear indicator if dataset is available or not
        dataset_status = "DATASET IS LOADED - Answer questions using the data above" if has_dataset else "NO DATASET LOADED - Use read_dataset tool first for data questions"
        
        return f"""You are a DataWarrior assistant for molecular data analysis.

STATUS: {dataset_status}

AVAILABLE TOOLS:
{tools_text}
{dataset_section}

=== STRICT RULES ===
1. Output ONLY valid JSON - no text before or after the JSON
2. Use ONLY the parameters listed above for each tool
3. DO NOT invent parameters that don't exist
4. Tools with "Parameters: NONE" must use empty arguments: {{}}
5. If dataset is shown above, answer directly WITHOUT calling tools
6. Use ACTUAL numbers from the data, never placeholders like [X] or [value]

=== RESPONSE FORMAT ===
Output this JSON structure ONLY:
{{
    "tool_calls": [
        {{"name": "tool_name", "arguments": {{}}, "reason": "why"}}
    ],
    "response": "Your answer to the user in their language"
}}

=== EXAMPLES ===

Example 1 - Data question WITHOUT dataset loaded:
User: "how many rows in the dataset?"
{{"tool_calls": [{{"name": "read_dataset", "arguments": {{}}, "reason": "Need data"}}], "response": "I am loading the data..."}}

Example 2 - Data question WITH dataset loaded (showing 50 rows, 5 columns):
User: "how many rows?"
{{"tool_calls": [], "response": "The dataset contains 50 rows and 5 columns."}}

Example 3 - Calculate descriptors:
User: "calculate logP and molecular weight"
{{"tool_calls": [{{"name": "calculate_descriptors", "arguments": {{"descriptors": ["logP", "totalWeight"]}}, "reason": "Calculate requested properties"}}], "response": "I am calculating logP and molecular weight."}}

Example 4 - Simple action:
User: "apply Lipinski filter"
{{"tool_calls": [{{"name": "apply_lipinski_filter", "arguments": {{}}, "reason": "Apply filter"}}], "response": "I am applying the Lipinski filter."}}

Example 5 - List columns WITH dataset:
User: "which columns are available?"
(Dataset shows columns: Name, MW, logP, TPSA)
{{"tool_calls": [], "response": "The available columns are: Name, MW, logP, TPSA."}}

=== CRITICAL REMINDERS ===
- read_dataset has NO parameters - always use {{}}
- calculate_descriptors requires: {{"descriptors": ["name1", "name2"]}}
- delete_columns requires: {{"columns": ["col1", "col2"]}}
- open_file requires: {{"filename": "file.sdf"}}
- Never output [X] or [value] - use real numbers from the dataset
- Output ONLY the JSON object, nothing else
"""
    
    def _parse_single_stage_response(self, response: str) -> OrchestratorResult:
        """Parses the Single-Stage response with better error tolerance"""
        try:
            clean = self._clean_json_response(response)
            data = json.loads(clean)
            
            tool_calls = []
            for tc in data.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    reason=tc.get("reason", "")
                ))
            
            user_response = data.get("response", "").strip()
            if not user_response:
                if tool_calls:
                    user_response = f"Executing {len(tool_calls)} action(s)..."
                else:
                    user_response = "How can I help you?"
            
            return OrchestratorResult(
                success=True,
                tool_calls=tool_calls,
                response=user_response,
                reasoning=""
            )
            
        except json.JSONDecodeError as e:
            # Try to extract JSON with regex
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    tool_calls = []
                    for tc in data.get("tool_calls", []):
                        tool_calls.append(ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                            reason=tc.get("reason", "")
                        ))
                    
                    return OrchestratorResult(
                        success=True,
                        tool_calls=tool_calls,
                        response=data.get("response", "Action in progress..."),
                        reasoning=""
                    )
                except:
                    pass
            
            # If no JSON but a text response
            clean_response = response.strip()
            if clean_response and not clean_response.startswith("{"):
                # It's probably a direct response
                return OrchestratorResult(
                    success=True,
                    tool_calls=[],
                    response=clean_response,
                    reasoning=""
                )
            
            return OrchestratorResult(
                success=True,
                tool_calls=[],
                response="I didn't understand well. Could you rephrase your request?",
                error=f"JSON parse error: {e}"
            )
    
    # ============================================
    # STAGE 3: RESPONSE GENERATION
    # ============================================
    
    async def generate_final_response(
        self, 
        user_message: str, 
        tool_results: List[Dict]
    ) -> str:
        """
        Stage 3: Generates the final response AFTER tool execution.
        Unified for large AND small models.
        """
        
        # Format tool results
        results_text = ""
        for tr in tool_results:
            tool_name = tr.get("tool_name", "unknown")
            status = tr.get("status", "unknown")
            
            if tool_name == "read_dataset" and status in ["success", "warning"]:
                data = tr.get("data", "")
                metadata = tr.get("metadata", {})
                results_text += f"\n[TOOL: {tool_name}]\n"
                results_text += f"Status: {status}\n"
                results_text += f"Rows: {metadata.get('rows', '?')}, Columns: {metadata.get('columns_count', '?')}\n"
                results_text += f"Column names: {', '.join(metadata.get('columns', []))}\n"
                results_text += f"Data:\n{data}\n"
            else:
                results_text += f"\n[TOOL: {tool_name}]\n"
                results_text += f"Status: {status}\n"
                results_text += f"Message: {tr.get('message', '')}\n"
        
        # Simplified prompt for small models
        if self.is_large_model:
            system_prompt = """You are a helpful assistant for DataWarrior, a cheminformatics platform.

You have just executed some tools and received their results. Now generate a helpful response for the user.

RULES:
1. Use the ACTUAL data from the tool results to answer
2. Be precise and accurate - use exact values from the data
3. Format the response nicely (use tables if appropriate)
4. Respond in the same language as the user
5. For numerical questions (max, min, count), calculate from the ACTUAL data provided

IMPORTANT: Only use information from the tool results below. Do NOT make up data."""
        else:
            # Simpler prompt for Ollama
            system_prompt = """Answer the user's question using ONLY the data provided below.
Use exact numbers from the data. Respond in the user's language.
Keep your answer concise and direct."""
        
        user_prompt = f"""User question: {user_message}

Tool results:
{results_text}

Answer the question using the actual data above:"""
        
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )
            
            # Add to appropriate history
            if self.is_large_model:
                self.planning_history.append({"role": "assistant", "content": response})
            else:
                self.conversation_history.append({"role": "assistant", "content": response})
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # ============================================
    # UTILITIES
    # ============================================
    
    def _clean_json_response(self, response: str) -> str:
        """Cleans the response to extract JSON"""
        clean = response.strip()
        
        # Remove markdown tags
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        
        clean = clean.strip()
        
        # Try to find JSON in the response
        if not clean.startswith("{"):
            # Search for the beginning of JSON
            start = clean.find("{")
            if start != -1:
                # Find the corresponding end
                depth = 0
                end = start
                for i, c in enumerate(clean[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                clean = clean[start:end]
        
        return clean.strip()
    
    def reset_history(self):
        """Resets all histories"""
        self.planning_history = []
        self.execution_history = []
        self.conversation_history = []
        self.current_dataset_context = None
        print("[Orchestrator] History and dataset context reset")
    
    def get_token_usage(self) -> Dict[str, int]:
        """Returns current token usage"""
        if self.is_large_model:
            history_tokens = estimate_messages_tokens(self.planning_history)
        else:
            history_tokens = estimate_messages_tokens(self.conversation_history)
        
        dataset_tokens = estimate_tokens(self.current_dataset_context or "")
        
        return {
            "history_tokens": history_tokens,
            "history_limit": self.limits.history_max_tokens,
            "dataset_tokens": dataset_tokens,
            "dataset_limit": self.limits.csv_warning_threshold,
            "total": history_tokens + dataset_tokens
        }


# ============================================
# FACTORY FUNCTIONS
# ============================================

def create_orchestrator_openai(api_key: str, model: str = "gpt-4o-mini") -> DataWarriorOrchestrator:
    """Creates an orchestrator with OpenAI (Two-Stage)"""
    client = create_openai_client(api_key, model)
    orchestrator = DataWarriorOrchestrator(client)
    orchestrator.set_compression_llm(api_key)
    return orchestrator


def create_orchestrator_anthropic(api_key: str, model: str = "claude-3-5-sonnet-20241022") -> DataWarriorOrchestrator:
    """Creates an orchestrator with Anthropic (Two-Stage)"""
    client = create_anthropic_client(api_key, model)
    return DataWarriorOrchestrator(client)


def create_orchestrator_ollama(model: str = "llama3.1:8b") -> DataWarriorOrchestrator:
    """Creates an orchestrator with Ollama (Single-Stage)"""
    client = create_ollama_client(model)
    return DataWarriorOrchestrator(client)
