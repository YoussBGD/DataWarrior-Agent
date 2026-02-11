#!/usr/bin/env python3
"""
DataWarrior-Agent  - Streamlit Interface
Chat + VNC + Upload + Dataset Context + Token Management

Fixes :
- Filtering of invalid arguments invented by small models
- Unified Stage 3 for all models (large AND small)
- Better error handling
"""

import streamlit as st
import asyncio
import json
from pathlib import Path

# Import MCP modules
from mcp_server import (
    open_file, save_as_csv, calculate_all_descriptors, calculate_descriptors,
    apply_lipinski_filter, delete_columns, generate_2d_coordinates,
    get_connection_status, get_session_info, get_available_descriptors,
    read_dataset
)

# Import LLM modules
from llm.orchestrator import (
    create_orchestrator_openai,
    create_orchestrator_anthropic, 
    create_orchestrator_ollama
)

# ============================================
# CONFIG
# ============================================

st.set_page_config(
    page_title="DataWarrior-Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="auto"
)

PROJECT_DIR = Path(__file__).parent
DATA_INPUT_DIR = PROJECT_DIR / "data" / "input"
CONFIG_FILE = PROJECT_DIR / ".config" / "api_keys.json"
NOVNC_URL = "http://localhost:6081/vnc.html?autoconnect=true&resize=scale&quality=9&password=datawarrior"

# ============================================
# VALID PARAMETERS FOR EACH TOOL 
# ============================================

TOOL_VALID_PARAMS = {
    "open_file": ["filename"],
    "save_as_csv": ["filename"],
    "calculate_all_descriptors": [],
    "calculate_descriptors": ["descriptors"],
    "apply_lipinski_filter": [],
    "delete_columns": ["columns"],
    "generate_2d_coordinates": [],
    "get_connection_status": [],
    "get_session_info": [],
    "get_available_descriptors": ["category"],
    "read_dataset": [],  # NO PARAMETERS!
    "list_available_tools": [],
}


def filter_arguments(tool_name: str, arguments: dict) -> dict:
    """
    Filters invalid arguments invented by the LLM.
    Small models like Llama sometimes invent parameters.
    """
    if not arguments:
        return {}
    
    valid_params = TOOL_VALID_PARAMS.get(tool_name, [])
    
    # If the tool has no valid parameters, return empty
    if not valid_params:
        if arguments:
            print(f"[Filter] {tool_name} takes no arguments, ignoring: {list(arguments.keys())}")
        return {}
    
    # Filter arguments
    filtered = {k: v for k, v in arguments.items() if k in valid_params}
    
    # Log removed arguments
    removed = set(arguments.keys()) - set(filtered.keys())
    if removed:
        print(f"[Filter] Removed invalid args for {tool_name}: {removed}")
    
    return filtered


# ============================================
# CSS STYLING
# ============================================

st.markdown("""
<style>
    /* Remove padding from main container */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        width: 320px;
        background-color: #1a1a2e !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #1a1a2e !important;
        padding: 1.5rem !important;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
        background-color: #16213e !important;
        color: white !important;
        border: 1px solid #0f3460 !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background-color: #e94560 !important;
        border-color: #e94560 !important;
    }
    
    /* Sidebar titles */
    section[data-testid="stSidebar"] h1 {
        color: #e94560 !important;
        font-size: 1.5rem !important;
        border-bottom: 2px solid #0f3460 !important;
        padding-bottom: 0.8rem !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00d9ff !important;
        font-size: 1.1rem !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #d0d0d0 !important;
    }
    
    /* Input fields */
    section[data-testid="stSidebar"] input {
        background-color: #16213e !important;
        border: 1px solid #0f3460 !important;
        color: white !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #00d9ff !important;
    }
    
    /* Warning box */
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Version badge */
    .version-badge {
        background-color: #e94560;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API KEY STORAGE
# ============================================

def load_api_keys():
    """Loads saved API keys"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_api_keys(keys: dict):
    """Saves API keys"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(keys, f)


# ============================================
# TOOL EXECUTOR WITH ARGUMENT FILTERING
# ============================================

# Tools that modify data and require a context refresh
DATA_MODIFYING_TOOLS = {
    "calculate_all_descriptors",
    "calculate_descriptors", 
    "apply_lipinski_filter",
    "delete_columns",
    "generate_2d_coordinates",
    "open_file",
    "save_as_csv"
}

def execute_tool(name: str, arguments: dict) -> dict:
    """
    Executes an MCP tool with filtering of invalid arguments.
    Manages the dataset context automatically.
    """
    
    # NEW: Filter invalid arguments
    filtered_arguments = filter_arguments(name, arguments)
    
    tools = {
        "open_file": open_file,
        "save_as_csv": save_as_csv,
        "calculate_all_descriptors": calculate_all_descriptors,
        "calculate_descriptors": calculate_descriptors,
        "apply_lipinski_filter": apply_lipinski_filter,
        "delete_columns": delete_columns,
        "generate_2d_coordinates": generate_2d_coordinates,
        "get_connection_status": get_connection_status,
        "get_session_info": get_session_info,
        "get_available_descriptors": get_available_descriptors,
        "read_dataset": read_dataset
    }
    
    if name not in tools:
        return {"status": "error", "message": f"Unknown tool: {name}"}
    
    try:
        # Execute the tool with filtered arguments
        if filtered_arguments:
            result = tools[name](**filtered_arguments)
        else:
            result = tools[name]()
        
        # Special handling for read_dataset
        if name == "read_dataset" and result["status"] in ["success", "warning"]:
            if st.session_state.orchestrator:
                # Inject the dataset into the orchestrator context
                context_warning = st.session_state.orchestrator.set_dataset_context(
                    result["data"],
                    result["metadata"]
                )
                
                if context_warning:
                    result["context_warning"] = context_warning
                
                # Simplified message for display (without the full CSV)
                result["display_message"] = (
                    f"Dataset loaded: {result['metadata'].get('rows', '?')} rows, "
                    f"{result['metadata'].get('columns_count', '?')} columns"
                )
        
        # AUTO-REFRESH: After tools that modify data
        elif name in DATA_MODIFYING_TOOLS and result.get("status") == "success":
            if st.session_state.orchestrator:
                import time
                time.sleep(1)
                
                # Read and inject the new dataset
                dataset_result = read_dataset()
                if dataset_result["status"] in ["success", "warning"]:
                    st.session_state.orchestrator.set_dataset_context(
                        dataset_result["data"],
                        dataset_result["metadata"]
                    )
                    result["context_updated"] = True
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# SESSION STATE
# ============================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "model_type" not in st.session_state:
    st.session_state.model_type = "ollama"
if "api_keys" not in st.session_state:
    st.session_state.api_keys = load_api_keys()
if "last_opened_file" not in st.session_state:
    st.session_state.last_opened_file = None


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown('<span class="version-badge">v1</span>', unsafe_allow_html=True)
    
    # ==================
    # Model Selection
    # ==================
    st.subheader("🤖 LLM Model")
    
    model_type = st.selectbox(
        "Provider",
        ["ollama", "openai", "anthropic"],
        index=0
    )
    
    if model_type == "ollama":
        model_name = st.text_input("Model", value="llama3.1:8b")
        
        if st.button("🔗 Connect Ollama", use_container_width=True):
            try:
                st.session_state.orchestrator = create_orchestrator_ollama(model_name)
                st.session_state.model_type = "ollama"
                st.success(f"✅ Connected to {model_name}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    elif model_type == "openai":
        saved_key = st.session_state.api_keys.get("openai", "")
        api_key = st.text_input("API Key", value=saved_key, type="password")
        save_key = st.checkbox("💾 Save API Key", value=bool(saved_key))
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        
        if st.button("🔗 Connect OpenAI", use_container_width=True) and api_key:
            try:
                st.session_state.orchestrator = create_orchestrator_openai(api_key, model_name)
                st.session_state.model_type = "openai"
                if save_key:
                    st.session_state.api_keys["openai"] = api_key
                else:
                    st.session_state.api_keys.pop("openai", None)
                save_api_keys(st.session_state.api_keys)
                st.success(f"✅ Connected to {model_name}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    elif model_type == "anthropic":
        saved_key = st.session_state.api_keys.get("anthropic", "")
        api_key = st.text_input("API Key", value=saved_key, type="password")
        save_key = st.checkbox("💾 Save API Key", value=bool(saved_key))
        model_name = st.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"])
        
        if st.button("🔗 Connect Claude", use_container_width=True) and api_key:
            try:
                st.session_state.orchestrator = create_orchestrator_anthropic(api_key, model_name)
                st.session_state.model_type = "anthropic"
                if save_key:
                    st.session_state.api_keys["anthropic"] = api_key
                else:
                    st.session_state.api_keys.pop("anthropic", None)
                save_api_keys(st.session_state.api_keys)
                st.success(f"✅ Connected to {model_name}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # ==================
    # Connection Status
    # ==================
    st.subheader("📌 DataWarrior")
    status = get_connection_status()
    if status["connected"]:
        st.success("✅ TCP Connected")
    else:
        st.error("❌ Not connected")
        st.caption("Start: Tools → Start Macro Agent")
    
    st.divider()
    
    # ==================
    # Token Usage
    # ==================
    st.subheader("📊 Token Usage")
    
    if st.session_state.orchestrator:
        try:
            usage = st.session_state.orchestrator.get_token_usage()
            
            # History tokens
            history_pct = min((usage["history_tokens"] / usage["history_limit"]) * 100, 100)
            st.markdown(f"**History:** {usage['history_tokens']:,} / {usage['history_limit']:,}")
            st.progress(history_pct / 100)
            
            if history_pct > 80:
                st.warning("⚠️ History near limit - will compress soon")
            
            # Dataset tokens
            if usage["dataset_tokens"] > 0:
                dataset_pct = min((usage["dataset_tokens"] / usage["dataset_limit"]) * 100, 100)
                st.markdown(f"**Dataset:** {usage['dataset_tokens']:,} / {usage['dataset_limit']:,}")
                st.progress(dataset_pct / 100)
                
                if dataset_pct > 80:
                    st.warning("⚠️ Large dataset - analysis may be incomplete")
            else:
                st.caption("No dataset in context")
            
            # Total
            st.caption(f"Total: {usage['total']:,} tokens")
            
        except Exception as e:
            st.caption("Token info unavailable")
    else:
        st.caption("Connect to LLM to see usage")
    
    st.divider()
    
    # ==================
    # Actions
    # ==================
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.orchestrator:
                st.session_state.orchestrator.reset_history()
            st.rerun()
    
    with col2:
        if st.button("📊 Clear Data", use_container_width=True):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.clear_dataset_context()
            st.success("Dataset context cleared")
            st.rerun()


# ============================================
# MAIN LAYOUT
# ============================================

# VNC View - DataWarrior display (full width)
st.components.v1.iframe(NOVNC_URL, height=950, scrolling=False)

# Bottom section: Upload + Chat
col_upload, col_chat = st.columns([1, 3])

# ==================
# Upload Section
# ==================
with col_upload:
    uploaded_file = st.file_uploader(
        "📁 Upload molecular file",
        type=["sdf", "mol", "csv", "txt"],
    )
    
    if uploaded_file:
        # Save file
        file_path = DATA_INPUT_DIR / uploaded_file.name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Auto-open only if new file
        if st.session_state.get("last_opened_file") != uploaded_file.name:
            # Clear old dataset context before opening new file
            if st.session_state.orchestrator:
                st.session_state.orchestrator.clear_dataset_context()
            
            # Delete old snapshot to avoid confusion
            snapshot_path = PROJECT_DIR / "data" / "output" / "_snapshot.csv"
            if snapshot_path.exists():
                snapshot_path.unlink()
            
            result = open_file(uploaded_file.name)
            if result["status"] == "success":
                st.session_state.last_opened_file = uploaded_file.name
                st.success(f"✅ Opened {uploaded_file.name}")
                st.caption("💡 Use 'read the dataset' or calculate descriptors to enable data queries")
            else:
                st.error(result["message"])
        else:
            st.success(f"✅ {uploaded_file.name} loaded")

# ==================
# Chat Section
# ==================
with col_chat:
    # Chat history container
    chat_container = st.container(height=260)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your molecules..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        if st.session_state.orchestrator is None:
            response = "⚠️ Please connect to an LLM first (open sidebar)"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Process with orchestrator
            with st.spinner("🧠 Thinking..."):
                try:
                    # Stage 1 & 2: Planning and Parameterization
                    result = asyncio.run(st.session_state.orchestrator.process(prompt))
                    
                    # Build response
                    full_response = ""
                    
                    # Add compression warning if present
                    if result.warning:
                        full_response += f"⚠️ {result.warning}\n\n"
                    
                    # Execute tool calls and collect results
                    if result.tool_calls:
                        tool_results = []
                        tool_outputs = []
                        
                        for tc in result.tool_calls:
                            # IMPORTANT: Arguments are filtered in execute_tool
                            tool_result = execute_tool(tc.name, tc.arguments)
                            
                            # Store full result for Stage 3
                            tool_result["tool_name"] = tc.name
                            tool_results.append(tool_result)
                            
                            # Prepare display message
                            if tc.name == "read_dataset":
                                display_msg = tool_result.get("display_message", tool_result.get("message", "Done"))
                            else:
                                display_msg = tool_result.get("message", "Done")
                            
                            status_icon = "✅" if tool_result["status"] != "error" else "❌"
                            tool_outputs.append(f"{status_icon} {tc.name}: {display_msg}")
                        
                        # ===== UNIFIED STAGE 3 =====
                        # Applied for ALL models (large AND small)
                        needs_stage3 = any(
                            tr.get("tool_name") == "read_dataset" and tr.get("status") in ["success", "warning"]
                            for tr in tool_results
                        )
                        
                        if needs_stage3:
                            # Generate response based on actual tool results
                            final_response = asyncio.run(
                                st.session_state.orchestrator.generate_final_response(prompt, tool_results)
                            )
                            full_response += final_response
                        else:
                            # Use Stage 1/Single-Stage response for non-data operations
                            full_response += result.response
                        
                        full_response += "\n\n**Executed:**\n" + "\n".join(tool_outputs)
                    else:
                        # No tools to execute, use direct response
                        full_response += result.response
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
