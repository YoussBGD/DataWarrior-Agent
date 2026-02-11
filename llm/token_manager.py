#!/usr/bin/env python3
"""
Token Management & History Compression for DataWarrior-Agent

Manages:
- Token count estimation
- History compression when limits are exceeded
- Reading and analyzing the CSV dataset
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


# ============================================
# LIMITS CONFIGURATION
# ============================================

@dataclass
class TokenLimits:
    """Token limits based on model type"""
    # CSV limits
    csv_warning_threshold: int      # Above this: warning
    csv_max_tokens: int             # Maximum absolute limit
    
    # History limits
    history_max_tokens: int         # Above this: compression
    history_compression_ratio: float  # Keep X% after compression
    summary_ratio: float            # Summary is X% of what is deleted


# Limits for large models (GPT-4, Claude)
LARGE_MODEL_LIMITS = TokenLimits(
    csv_warning_threshold=30000,
    csv_max_tokens=50000,
    history_max_tokens=100000,
    history_compression_ratio=0.30,  # Keep 30%
    summary_ratio=0.10               # Summary = 10% of the 70% deleted
)

# Limits for small models (Ollama)
SMALL_MODEL_LIMITS = TokenLimits(
    csv_warning_threshold=4000,
    csv_max_tokens=6000,
    history_max_tokens=15000,
    history_compression_ratio=0.30,
    summary_ratio=0.10
)


# ============================================
# TOKEN ESTIMATION
# ============================================

def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in a text.
    Approximate rule: ~1 token = 4 characters
    """
    if not text:
        return 0
    return len(text) // 4


def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimates tokens for a list of messages"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # Overhead for role, etc.
    return total


# ============================================
# DATASET READING
# ============================================

def read_dataset_csv(csv_path: Path, limits: TokenLimits) -> Dict:
    """
    Reads a CSV file and returns its content with limit management.
    
    Returns:
        {
            "status": "success" | "warning" | "error",
            "message": str,
            "data": str (CSV content),
            "metadata": {
                "rows": int,
                "columns": list,
                "tokens": int,
                "truncated": bool
            }
        }
    """
    if not csv_path.exists():
        return {
            "status": "error",
            "message": f"Dataset file not found: {csv_path}",
            "data": "",
            "metadata": {}
        }
    
    try:
        # Read the file
        content = csv_path.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        if not lines:
            return {
                "status": "error",
                "message": "Dataset is empty",
                "data": "",
                "metadata": {}
            }
        
        # Extract metadata
        headers = lines[0].split(',')
        num_rows = len(lines) - 1  # Minus the header
        
        # Estimate tokens
        tokens = estimate_tokens(content)
        
        # Prepare metadata
        metadata = {
            "rows": num_rows,
            "columns": headers,
            "columns_count": len(headers),
            "tokens": tokens,
            "truncated": False
        }
        
        # Check limits
        if tokens > limits.csv_max_tokens:
            # Truncate the content
            truncated_content = _truncate_csv(lines, limits.csv_max_tokens)
            metadata["truncated"] = True
            metadata["original_tokens"] = tokens
            metadata["tokens"] = estimate_tokens(truncated_content)
            
            return {
                "status": "warning",
                "message": f"⚠️ Very large dataset ({tokens} tokens). Truncated to {metadata['tokens']} tokens. Some data may not be analyzed.",
                "data": truncated_content,
                "metadata": metadata
            }
        
        elif tokens > limits.csv_warning_threshold:
            return {
                "status": "warning", 
                "message": f"⚠️ Large dataset ({tokens} tokens). Analysis may be incomplete due to context limitations.",
                "data": content,
                "metadata": metadata
            }
        
        else:
            return {
                "status": "success",
                "message": f"Dataset loaded: {num_rows} molecules, {len(headers)} columns",
                "data": content,
                "metadata": metadata
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading dataset: {str(e)}",
            "data": "",
            "metadata": {}
        }


def _truncate_csv(lines: List[str], max_tokens: int) -> str:
    """Truncates a CSV to respect the token limit"""
    header = lines[0]
    result_lines = [header]
    current_tokens = estimate_tokens(header)
    
    for line in lines[1:]:
        line_tokens = estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens:
            break
        result_lines.append(line)
        current_tokens += line_tokens
    
    # Add a truncation indicator
    result_lines.append(f"... [TRUNCATED - showing {len(result_lines)-1} of {len(lines)-1} rows]")
    
    return '\n'.join(result_lines)


# ============================================
# HISTORY COMPRESSION
# ============================================

async def compress_history(
    messages: List[Dict[str, str]], 
    limits: TokenLimits,
    llm_client  # LLMClient for the summary
) -> Tuple[List[Dict[str, str]], str]:
    """
    Compresses history when it exceeds limits.
    
    Process:
    1. Calculate how much to delete (70%)
    2. Summarize the deleted messages (summary = 10% of what is deleted)
    3. Keep the recent 30%
    
    Returns:
        (new_messages, compression_message)
    """
    total_tokens = estimate_messages_tokens(messages)
    
    if total_tokens <= limits.history_max_tokens:
        return messages, ""
    
    print(f"[Compression] History exceeds limit: {total_tokens} > {limits.history_max_tokens}")
    
    # Calculate the cutoff point (keep 30% of the most recent)
    keep_ratio = limits.history_compression_ratio
    keep_count = max(2, int(len(messages) * keep_ratio))  # Keep at least 2 messages
    
    messages_to_remove = messages[:-keep_count]
    messages_to_keep = messages[-keep_count:]
    
    print(f"[Compression] Removing {len(messages_to_remove)} messages, keeping {len(messages_to_keep)}")
    
    # Create a summary of the deleted messages
    summary = await _create_history_summary(messages_to_remove, limits, llm_client)
    
    # Build the new history
    summary_message = {
        "role": "system",
        "content": f"[CONVERSATION HISTORY SUMMARY]\n{summary}\n[END SUMMARY - Recent messages follow]"
    }
    
    new_messages = [summary_message] + messages_to_keep
    
    new_tokens = estimate_messages_tokens(new_messages)
    compression_msg = f"🗜️ History compressed: {total_tokens} → {new_tokens} tokens ({len(messages)} → {len(new_messages)} messages)"
    
    print(f"[Compression] {compression_msg}")
    
    return new_messages, compression_msg


async def _create_history_summary(
    messages: List[Dict[str, str]], 
    limits: TokenLimits,
    llm_client
) -> str:
    """
    Creates a summary of messages with an LLM.
    The summary should be ~10% of the original size.
    """
    # Calculate target summary size
    original_tokens = estimate_messages_tokens(messages)
    target_tokens = int(original_tokens * limits.summary_ratio)
    target_tokens = max(100, min(target_tokens, 500))  # Between 100 and 500 tokens
    
    # Prepare the content to summarize
    conversation_text = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"
    
    # Prompt for the summary
    summary_prompt = f"""Summarize this conversation history concisely.

RULES:
- Keep only the essential information: what actions were requested and executed
- Do NOT include CSV data or detailed results
- Focus on: files opened, operations performed, key user requests
- Target length: ~{target_tokens} tokens (be concise)
- Write in the same language as the conversation

CONVERSATION TO SUMMARIZE:
{conversation_text}

SUMMARY:"""

    try:
        # Use the LLM to summarize
        response = await llm_client.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            system_prompt="You are a conversation summarizer. Be concise and factual."
        )
        return response.strip()
        
    except Exception as e:
        # Fallback: basic summary
        print(f"[Compression] LLM summary failed: {e}, using fallback")
        return _fallback_summary(messages)


def _fallback_summary(messages: List[Dict[str, str]]) -> str:
    """Basic summary if LLM fails"""
    user_messages = [m for m in messages if m["role"] == "user"]
    
    if not user_messages:
        return "Previous conversation with no specific user requests."
    
    # Extract the first requests
    requests = []
    for msg in user_messages[:5]:  # Max 5 first requests
        content = msg["content"][:100]  # Truncate
        requests.append(f"- {content}")
    
    summary = "Previous conversation included:\n" + "\n".join(requests)
    
    if len(user_messages) > 5:
        summary += f"\n... and {len(user_messages) - 5} more exchanges."
    
    return summary


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_limits_for_model(is_large_model: bool) -> TokenLimits:
    """Returns the appropriate limits based on the model"""
    return LARGE_MODEL_LIMITS if is_large_model else SMALL_MODEL_LIMITS


def format_dataset_for_llm(dataset_result: Dict) -> str:
    """
    Formats the result of read_dataset_csv for inclusion in the LLM context.
    """
    if dataset_result["status"] == "error":
        return f"[ERROR] {dataset_result['message']}"
    
    metadata = dataset_result["metadata"]
    
    header = f"""[CURRENT DATASET]
Rows: {metadata.get('rows', 'unknown')}
Columns: {', '.join(metadata.get('columns', []))}
Tokens: {metadata.get('tokens', 'unknown')}
"""
    
    if dataset_result["status"] == "warning":
        header += f"⚠️ {dataset_result['message']}\n"
    
    header += "\n[DATA]\n"
    header += dataset_result["data"]
    header += "\n[END DATASET]"
    
    return header
