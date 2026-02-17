"""
Test script to verify LLM provider connections.
Priority: OpenAI > HuggingFace > Ollama
"""
import os
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env file from: {env_path}")
else:
    print(f"[INFO] .env file not found at {env_path}")
    print("       Using environment variables from system")

from src.llm.providers import get_llm_provider, OllamaProvider, HuggingFaceProvider, OpenAIProvider


def test_ollama():
    """Test Ollama connection."""
    print("="*60)
    print("Testing Ollama Connection")
    print("="*60)
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3:8b")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print()
    
    try:
        provider = OllamaProvider(base_url=base_url, model=model)
        
        if not provider._available:
            print("[FAIL] Ollama is not running or not accessible.")
            print("       Make sure Ollama is installed and running:")
            print("       1. Install from https://ollama.ai/download")
            print("       2. Run: ollama pull llama3:8b")
            print("       3. Start Ollama service")
            return False
        
        print("[OK] Ollama is accessible")
        print("\nTesting generation...")
        
        response = provider.generate(
            "What is COPD? Answer in one sentence.",
            max_tokens=50
        )
        
        print(f"[SUCCESS] Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_huggingface():
    """Test HuggingFace connection."""
    print("\n" + "="*60)
    print("Testing HuggingFace Connection")
    print("="*60)
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    model = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    
    if not api_key:
        print("[SKIP] HUGGINGFACE_API_KEY not set")
        print("       Set it with: $env:HUGGINGFACE_API_KEY='your_key'")
        return None
    
    print(f"Model: {model}")
    print()
    
    try:
        provider = HuggingFaceProvider(api_key=api_key, model=model)
        
        print("[OK] HuggingFace provider initialized")
        print("\nTesting generation...")
        
        response = provider.generate(
            "What is COPD? Answer in one sentence.",
            max_tokens=50
        )
        
        print(f"[SUCCESS] Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_openai():
    """Test OpenAI connection."""
    print("\n" + "="*60)
    print("Testing OpenAI Connection")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    if not api_key:
        print("[SKIP] OPENAI_API_KEY not set")
        print("       Set it in .env file: OPENAI_API_KEY=sk-...")
        return None
    
    print(f"Model: {model}")
    print()
    
    try:
        provider = OpenAIProvider(api_key=api_key, model=model)
        
        if not provider._available:
            print("[FAIL] OpenAI client not available")
            print("       Check if 'openai' package is installed: pip install openai")
            return False
        
        print("[OK] OpenAI provider initialized")
        print("\nTesting generation...")
        
        response = provider.generate(
            "What is COPD? Answer in one sentence.",
            max_tokens=50
        )
        
        print(f"[SUCCESS] Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_auto_detect():
    """Test auto-detection of LLM provider."""
    print("\n" + "="*60)
    print("Testing Auto-Detection")
    print("="*60)
    
    from src.llm.providers import get_llm_provider
    
    provider = get_llm_provider()
    
    if provider:
        print(f"[SUCCESS] Auto-detected provider: {type(provider).__name__}")
        
        # Test generation
        try:
            response = provider.generate(
                "What is COPD? Answer in one sentence.",
                max_tokens=50
            )
            print(f"[SUCCESS] Test generation works: {response[:50]}...")
            return True
        except Exception as e:
            print(f"[WARNING] Generation failed: {e}")
            return False
    else:
        print("[FAIL] No LLM provider available")
        print("\nTo set up HuggingFace (recommended):")
        print("1. Get API token from https://huggingface.co/settings/tokens")
        print("2. Set: $env:HUGGINGFACE_API_KEY='your_token'")
        print("3. Optional: Set $env:HUGGINGFACE_MODEL='model_name'")
        return False


def main():
    """Run all LLM connection tests. Priority: Ollama > HuggingFace > OpenAI."""
    print("\n" + "="*60)
    print("LLM Connection Test Suite")
    print("="*60)
    print("\nPriority: Ollama (Local) > HuggingFace > OpenAI")
    print()
    
    results = []
    
    # Test Ollama first (local, free, preferred)
    results.append(("Ollama", test_ollama()))
    
    # Test HuggingFace (kept for future use)
    results.append(("HuggingFace", test_huggingface()))
    
    # Test OpenAI (optional fallback)
    results.append(("OpenAI", test_openai()))
    
    # Test auto-detection
    print("\n" + "="*60)
    print("Testing Auto-Detection")
    print("="*60)
    
    provider = get_llm_provider()  # Auto-detect
    if provider:
        print(f"[SUCCESS] Auto-detected provider: {type(provider).__name__}")
        try:
            response = provider.generate("Test", max_tokens=10)
            print(f"[SUCCESS] Generation works!")
        except Exception as e:
            print(f"[WARNING] Generation failed: {e}")
    else:
        print("[FAIL] No providers available")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        if result is None:
            status = "[SKIP]"
        elif result:
            status = "[OK]"
        else:
            status = "[FAIL]"
        print(f"{status} {name}")
    
    any_working = any(r for r in results if r is True)
    
    if any_working:
        print("\n[SUCCESS] At least one LLM provider is working!")
        print("You can now use LLM-based predictions.")
    else:
        print("\n[WARNING] No LLM providers are working.")
        print("The system will use fallback rule-based predictions.")
        print("\n" + "="*60)
        print("RECOMMENDED: Set up Ollama (Local, Free)")
        print("="*60)
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull models:")
        print("   ollama pull mistral:7b-instruct-q4_0")
        print("   ollama pull llama3:8b")
        print("3. Ensure Ollama is running (should auto-start)")
        print("4. Test: ollama run mistral:7b-instruct-q4_0")
        print("\nNote: HuggingFace free models are deprecated.")
        print("      OpenAI requires credits/billing.")


if __name__ == "__main__":
    main()
