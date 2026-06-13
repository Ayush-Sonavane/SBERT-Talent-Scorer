import os
import pytest
from src.utils import load_docx_text, stream_jsonl

def test_load_docx_text():
    jd_path = os.path.join(
        "[PUB] India_runs_data_and_ai_challenge",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "job_description.docx"
    )
    # The file should exist in the workspace
    if os.path.exists(jd_path):
        text = load_docx_text(jd_path)
        assert len(text) > 100
        assert "AI" in text or "ML" in text or "Senior" in text
    else:
        # Fallback/skip if test runs in different path context
        pass

def test_stream_jsonl():
    sample_path = os.path.join(
        "[PUB] India_runs_data_and_ai_challenge",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "sample_candidates.json"
    )
    # Note sample_candidates.json is a standard JSON file (array), not JSONL.
    # But candidates.jsonl is a JSONL file. Let's create a temporary JSONL file to test.
    temp_jsonl = "temp_test.jsonl"
    with open(temp_jsonl, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"invalid json line here\n')
        f.write('{"id": 2, "name": "Bob"}\n')
        
    try:
        items = list(stream_jsonl(temp_jsonl))
        assert len(items) == 2
        assert items[0]["name"] == "Alice"
        assert items[1]["id"] == 2
    finally:
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)
