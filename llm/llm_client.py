#!/usr/bin/env python3
"""
Unified LLM Client for GPT / Claude / Ollama
"""

import os
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    provider: ModelProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama: http://localhost:11434
    temperature: float = 0.3
    max_tokens: int = 2000


class LLMClient:
    """Unified client for different LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initializes the client based on the provider"""
        if self.config.provider == ModelProvider.OPENAI:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.config.api_key)
        
        elif self.config.provider == ModelProvider.ANTHROPIC:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.config.api_key)
        
        elif self.config.provider == ModelProvider.OLLAMA:
            # Ollama uses a simple REST API
            self.config.base_url = self.config.base_url or "http://localhost:11434"
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Sends a message and returns the response
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        if self.config.provider == ModelProvider.OPENAI:
            return await self._chat_openai(messages, system_prompt)
        
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return await self._chat_anthropic(messages, system_prompt)
        
        elif self.config.provider == ModelProvider.OLLAMA:
            return await self._chat_ollama(messages, system_prompt)
    
    async def _chat_openai(self, messages: List[Dict], system_prompt: Optional[str]) -> str:
        """Chat with OpenAI GPT"""
        full_messages = []
        
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        
        full_messages.extend(messages)
        
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def _chat_anthropic(self, messages: List[Dict], system_prompt: Optional[str]) -> str:
        """Chat with Anthropic Claude"""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt or "",
            messages=messages
        )
        
        return response.content[0].text
    
    async def _chat_ollama(self, messages: List[Dict], system_prompt: Optional[str]) -> str:
        """Chat with Ollama (local)"""
        import aiohttp
        
        # Build the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n"
        
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n\n"
        
        full_prompt += "Assistant: "
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            async with session.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            ) as resp:
                data = await resp.json()
                return data.get("response", "")
    
    def chat_sync(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """Synchronous version of chat()"""
        import asyncio
        return asyncio.run(self.chat(messages, system_prompt))
    
    @staticmethod
    def is_large_model(model: str) -> bool:
        """Determines if it's a large model (for 2-stages) or small (semantic search)"""
        large_models = [
            "gpt-4", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-opus", "claude-3-sonnet", "claude-3-5-sonnet",
            "llama3.1:70b", "llama3.1:405b"
        ]
        
        for large in large_models:
            if large in model.lower():
                return True
        
        return False


# ============================================
# Factory functions
# ============================================

def create_openai_client(api_key: str, model: str = "gpt-4o-mini") -> LLMClient:
    """Creates an OpenAI client"""
    config = LLMConfig(
        provider=ModelProvider.OPENAI,
        model=model,
        api_key=api_key
    )
    return LLMClient(config)


def create_anthropic_client(api_key: str, model: str = "claude-3-5-sonnet-20241022") -> LLMClient:
    """Creates an Anthropic client"""
    config = LLMConfig(
        provider=ModelProvider.ANTHROPIC,
        model=model,
        api_key=api_key
    )
    return LLMClient(config)


def create_ollama_client(model: str = "llama3.1:8b") -> LLMClient:
    """Creates an Ollama client (local, no API key)"""
    config = LLMConfig(
        provider=ModelProvider.OLLAMA,
        model=model,
        base_url="http://localhost:11434"
    )
    return LLMClient(config)
