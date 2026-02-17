"""
Quick script to check if .env file is being loaded correctly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Check .env file location
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

print("="*60)
print("Environment Variable Check")
print("="*60)
print(f"\nProject root: {project_root}")
print(f".env file path: {env_path}")
print(f".env file exists: {env_path.exists()}")

if env_path.exists():
    print("\nLoading .env file...")
    load_dotenv(env_path)
    print("[OK] .env file loaded")
    
    # Check contents
    print("\n.env file contents:")
    print("-" * 60)
    with env_path.open("r", encoding="utf-8") as f:
        content = f.read()
        # Hide actual token for security
        safe_content = content.replace(os.getenv("HUGGINGFACE_API_KEY", ""), "hf_***HIDDEN***")
        print(safe_content)
    print("-" * 60)
else:
    print("\n[WARNING] .env file not found!")
    print(f"Create it at: {env_path}")
    print("\nContents should be:")
    print("HUGGINGFACE_API_KEY=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")
    print("HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2")

# Check environment variables
print("\n" + "="*60)
print("Environment Variables")
print("="*60)

hf_key = os.getenv("HUGGINGFACE_API_KEY")
hf_model = os.getenv("HUGGINGFACE_MODEL")
import os
token = os.getenv("HF_TOKEN")

if hf_key:
    # Show first and last few chars for verification
    masked_key = f"{hf_key[:5]}...{hf_key[-5:]}" if len(hf_key) > 10 else "***"
    print(f"[OK] HUGGINGFACE_API_KEY: {masked_key}")
else:
    print("[FAIL] HUGGINGFACE_API_KEY: Not set")

if hf_model:
    print(f"[OK] HUGGINGFACE_MODEL: {hf_model}")
else:
    print("[FAIL] HUGGINGFACE_MODEL: Not set (using default)")

print("\n" + "="*60)
if hf_key:
    print("[SUCCESS] HuggingFace API key is set!")
    print("You can now run: python -m scripts.test_llm_connection")
else:
    print("[FAIL] HuggingFace API key is NOT set")
    print("\nTo fix:")
    print(f"1. Create .env file at: {env_path}")
    print("2. Add: HUGGINGFACE_API_KEY=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")
    print("3. Add: HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2")
print("="*60)
