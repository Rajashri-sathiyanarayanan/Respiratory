"""
Test script to verify SQLite database integration for patients and conversation history.
"""
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.db import (
    init_db,
    get_connection,
    get_or_create_session,
    get_or_create_patient,
    save_conversation_turn,
    get_conversation_history,
    format_history_for_context,
)
from src.config import PATHS


def test_db_init():
    """Test database initialization."""
    print("=" * 60)
    print("1. Testing database initialization...")
    print("=" * 60)
    init_db()
    db_path = PATHS.db_path
    assert db_path.exists(), f"DB file not created at {db_path}"
    print(f"   [OK] Database created at: {db_path}")
    print()


def test_schema():
    """Test that tables exist."""
    print("=" * 60)
    print("2. Testing schema (tables exist)...")
    print("=" * 60)
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        expected = {"patients", "sessions", "conversations"}
        for t in expected:
            assert t in tables, f"Table '{t}' not found"
        print(f"   [OK] Tables found: {tables}")
    finally:
        conn.close()
    print()


def test_conversation_flow():
    """Test saving and loading conversation turns."""
    print("=" * 60)
    print("3. Testing conversation save/load...")
    print("=" * 60)

    # Create session
    session_id = get_or_create_session()
    print(f"   Session ID: {session_id}")

    # Save user message
    save_conversation_turn(session_id, "user", "What is COPD?")
    print("   [OK] Saved user message")

    # Save assistant answer
    save_conversation_turn(
        session_id,
        "assistant",
        "COPD is chronic obstructive pulmonary disease...",
        evidence_ids=["ev1", "ev2"],
        llm_model="mistral:7b",
    )
    print("   [OK] Saved assistant answer")

    # Load history
    history = get_conversation_history(session_id, limit=10)
    assert len(history) == 2, f"Expected 2 turns, got {len(history)}"
    assert history[0][0] == "user"
    assert history[1][0] == "assistant"
    print(f"   [OK] Loaded {len(history)} conversation turns")

    # Format for context
    context = format_history_for_context(history)
    assert "Previous conversation:" in context
    assert "User" in context
    assert "Assistant" in context
    print("   [OK] Context formatting works")
    print()


def test_patient_creation():
    """Test patient record creation."""
    print("=" * 60)
    print("4. Testing patient creation...")
    print("=" * 60)

    patient_id = get_or_create_patient(
        age_range="50-60",
        sex="M",
        smoking_status="former",
        pack_years="20",
        comorbidities=["hypertension"],
    )
    print(f"   Patient ID: {patient_id}")

    session_id = get_or_create_session(patient_id=patient_id)
    print(f"   Session linked to patient: {session_id}")
    print("   [OK] Patient and session creation works")
    print()


def main():
    """Run all DB integration tests."""
    print("\n" + "=" * 60)
    print("DB INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    test_db_init()
    test_schema()
    test_conversation_flow()
    test_patient_creation()

    print("=" * 60)
    print("[SUCCESS] All DB integration tests passed!")
    print("=" * 60)
    print("\nNext: Use session_id in /ask and /chat requests for context-aware answers.")
    print("The API will save prompts and answers to the database automatically.")


if __name__ == "__main__":
    main()
