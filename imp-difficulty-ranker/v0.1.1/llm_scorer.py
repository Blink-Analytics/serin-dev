"""
LLM Scorer Module
Handles API calls to Groq for importance/difficulty scoring with retry logic and validation.
"""

import json
import time
from typing import Dict, Optional, Tuple
import requests

from obj_sys_prompt import generate_interview_scoring_prompt_v1


class LLMScorer:
    """Handles scoring requests to LLM providers with retry logic and validation."""
    
    def __init__(self, provider: str, model: str, api_key: str):
        """
        Initialize the LLM scorer.
        
        Args:
            provider: The LLM provider ('groq' for now)
            model: The model name (e.g., 'llama-3.3-70b-versatile')
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = self._get_base_url()
        
    def _get_base_url(self) -> str:
        """Get the base URL for the provider."""
        if self.provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def score_objective(
        self, 
        job_description_json: str, 
        target_objective: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Score a single objective using the LLM.
        
        Args:
            job_description_json: The job description (can be JSON or extracted text)
            target_objective: The objective to score
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict with keys: importance, difficulty, response_time_ms, error
        """
        start_time = time.time()
        
        # Generate the prompt
        prompt = generate_interview_scoring_prompt_v1(job_description_json, target_objective)
        
        # Try multiple times
        for attempt in range(max_retries):
            try:
                response = self._call_api(prompt)
                importance, difficulty = self._parse_response(response)
                
                # Calculate response time
                response_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "importance": importance,
                    "difficulty": difficulty,
                    "response_time_ms": response_time_ms,
                    "error": None
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    # Extract wait time from error message if available
                    if "try again in" in error_msg:
                        try:
                            # Parse wait time (e.g., "659.999999ms" or "2.5s")
                            import re
                            match = re.search(r'try again in ([\d.]+)(ms|s)', error_msg)
                            if match:
                                wait_time = float(match.group(1))
                                unit = match.group(2)
                                if unit == "ms":
                                    wait_time = wait_time / 1000
                                # Add buffer
                                wait_time += 1
                                time.sleep(wait_time)
                            else:
                                time.sleep(5)  # Default 5 seconds
                        except:
                            time.sleep(5)
                    else:
                        time.sleep(5)  # Default 5 seconds for rate limit
                else:
                    # Regular error - exponential backoff
                    time.sleep(2 ** attempt)
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    response_time_ms = int((time.time() - start_time) * 1000)
                    return {
                        "importance": -1,
                        "difficulty": -1,
                        "response_time_ms": response_time_ms,
                        "error": error_msg
                    }
        
        # Should never reach here
        return {
            "importance": -1,
            "difficulty": -1,
            "response_time_ms": 0,
            "error": "Unknown error"
        }
    
    def _call_api(self, prompt: str) -> str:
        """
        Make API call to the LLM provider.
        
        Args:
            prompt: The system prompt to send
            
        Returns:
            Raw response string from the LLM
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 300,  # Increased to handle batch scoring (5+ objectives)
            "response_format": {"type": "json_object"}  # Force JSON mode
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _parse_response(self, response: str) -> Tuple[int, int]:
        """
        Parse and validate the LLM response.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Tuple of (importance, difficulty) as floats rounded to 1 decimal
            
        Raises:
            ValueError if response is invalid
        """
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Validate keys
            if "importance" not in data or "difficulty" not in data:
                raise ValueError(f"Missing keys in response: {data}")
            
            importance = data["importance"]
            difficulty = data["difficulty"]
            
            # Validate types (accept int or float)
            if not isinstance(importance, (int, float)) or not isinstance(difficulty, (int, float)):
                raise ValueError(f"Non-numeric values: importance={importance}, difficulty={difficulty}")
            
            # Convert to float and round to 1 decimal place
            importance = round(float(importance), 1)
            difficulty = round(float(difficulty), 1)
            
            # Validate range (clamp to 0.0-10.0)
            importance = max(0.0, min(10.0, importance))
            difficulty = max(0.0, min(10.0, difficulty))
            
            return importance, difficulty
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {response}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse response: {response}") from e


def test_scorer():
    """Test function to verify the scorer works."""
    import os
    
    # Example usage
    api_key = os.environ.get("GROQ_API_KEY", "your-api-key-here")
    
    scorer = LLMScorer(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=api_key
    )
    
    test_jd = """{
        "jobProfile": {
            "JobID": "BE-001",
            "coreDetails": {
                "title": "Senior Backend Engineer",
                "jobSummary": "Build scalable microservices"
            }
        }
    }"""
    
    test_obj = "Design distributed system architecture"
    
    result = scorer.score_objective(test_jd, test_obj)
    print("Test Result:")
    print(f"  Importance: {result['importance']}")
    print(f"  Difficulty: {result['difficulty']}")
    print(f"  Response Time: {result['response_time_ms']}ms")
    print(f"  Error: {result['error']}")


if __name__ == "__main__":
    test_scorer()
