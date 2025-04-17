from typing import Optional
import os
import requests


class PerplexityClient:
    """
    Client for the Perplexity API.
    """

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.perplexity.ai/chat/completions"):
        if api_key is None:
            api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("The Perplexity API key was not provided")
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query: str, model: str = "sonar-pro", max_tokens: int = 300) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": query}
            ],
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        _response = requests.post(self.base_url, json=payload, headers=headers)
        _result = _response.json()
        return _result["choices"][0]["message"]["content"]
