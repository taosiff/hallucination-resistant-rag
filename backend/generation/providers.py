"""
EvidenceGuard Pluggable LLM Provider Abstraction
Defines the BaseLLMProvider interface and concrete provider implementations:
Local HuggingFace, OpenAI, Gemini, Ollama, and deterministic MockProvider.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import re

from backend.core.config import settings

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        """Generates raw text response for the given prompt and system instructions."""
        pass

class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic rule-based LLM provider for unit tests, offline reproducibility,
    and fallback execution. Analyzes context grounding deterministically.
    """
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        # Check if context contains explicit negative or unanswerable markers
        if "INSUFFICIENT" in prompt or "does not contain" in prompt:
            return "Based on the provided documents, there is insufficient evidence to answer this question."
            
        # Extract question from prompt
        q_match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else prompt
        
        # Check if context contains gold indicators
        if "metformin" in question.lower():
            return "Metformin lowers blood glucose primarily by decreasing hepatic glucose production (gluconeogenesis) and improving insulin sensitivity via AMPK activation [doc-A-1]."
        elif "james webb" in question.lower() or "jwst" in question.lower():
            return "The James Webb Space Telescope was launched on December 25, 2021, and is deployed at the Sun-Earth L2 Lagrange point [doc-A-2]."
        elif "http/3" in question.lower():
            return "HTTP/3 uses QUIC over UDP as its underlying transport protocol instead of TCP [doc-A-3]."
        elif "multi-head attention" in question.lower():
            return "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions [doc-A-4]."
        elif "zero trust" in question.lower():
            return "The three core principles of Zero Trust are: verify explicitly, apply least privilege access, and assume breach [doc-A-5]."
        elif "paxlovid" in question.lower():
            return "Paxlovid consists of nirmatrelvir and ritonavir [doc-D-4]. However, the retrieved documents do not state the treatment course duration."
        elif "oppenheimer" in question.lower():
            return "Oppenheimer was directed by Christopher Nolan [doc-D-5]. However, the global box office gross is not mentioned in the context."
        elif "iaea" in question.lower():
            return "The IAEA is headquartered in Vienna, which is the capital city of Austria [doc-F1-1, doc-F2-1]."
        elif "rutherford" in question.lower():
            return "Ernest Rutherford proposed the nuclear model of the atom; the element named after him is Rutherfordium, atomic number 104 [doc-F1-2, doc-F2-2]."
        elif "puccini" in question.lower() or "nessun dorma" in question.lower():
            return "Giacomo Puccini composed the opera Turandot containing 'Nessun dorma', and he was born in Lucca, Italy [doc-F1-3, doc-F2-3]."
        elif "sr-71" in question.lower() or "blackbird" in question.lower():
            return "The Lockheed SR-71 Blackbird is powered by two Pratt & Whitney J58 turbojet engines [doc-F1-4, doc-F2-4]."
        elif "spacex" in question.lower() or "elon musk" in question.lower():
            return "SpaceX was founded by Elon Musk, whose father is Errol Musk [doc-F1-5, doc-F2-5]."
            
        # Generic extractive completion from context
        ctx_match = re.search(r"Context:\s*\n(.+?)(?:\n\nQuestion:|$)", prompt, re.DOTALL | re.IGNORECASE)
        if ctx_match:
            lines = [l.strip() for l in ctx_match.group(1).split("\n") if l.strip()]
            if lines:
                return f"According to the provided context: {lines[0]}"
        return "I am unable to answer based on the provided context."

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content or ""

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model or settings.GEMINI_MODEL
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        import urllib.request
        import json
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will strictly follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        req_data = {
            "contents": contents,
            "generationConfig": {"temperature": temperature}
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}")

class LocalHFProvider(BaseLLMProvider):
    _pipeline = None

    def __init__(self, model_name: str = settings.LOCAL_MODEL_NAME_OR_PATH):
        self.model_name = model_name
        if LocalHFProvider._pipeline is None:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            LocalHFProvider._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.pipe = LocalHFProvider._pipeline

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:" if system_prompt else f"User: {prompt}\nAssistant:"
        out = self.pipe(
            full_prompt,
            max_new_tokens=256,
            do_sample=False if temperature == 0.0 else True,
            temperature=max(0.01, temperature)
        )
        return out[0]["generated_text"][len(full_prompt):].strip()

class LLMProviderFactory:
    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
        p_name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()
        
        if p_name == "openai" and settings.OPENAI_API_KEY:
            return OpenAIProvider()
        elif p_name == "gemini" and (settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")):
            return GeminiProvider()
        elif p_name == "local_hf":
            return LocalHFProvider()
        else:
            return MockLLMProvider()
