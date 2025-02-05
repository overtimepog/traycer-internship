import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio
import json
from cli import display_final_plan  # Updated import

def test_display_codebase_analysis():
    # Construct a dummy plan with only codebase_analysis content.
    dummy_plan = {
        "explanation": "Test explanation for codebase analysis.",
        "files_modified": [],
        "codebase_analysis": {
            "Current Implementation": ["File scanning using regex", "Sequential processing"],
            "Detailed Analysis": ["Low caching", "Insufficient parallelism"]
        }
    }
    # Pass the dummy plan as a JSON string.
    asyncio.run(display_final_plan(json.dumps(dummy_plan)))

if __name__ == "__main__":
    test_display_codebase_analysis()
