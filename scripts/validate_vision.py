#!/usr/bin/env python3
"""
Vision Validation Script

Runs all validation tests and produces a report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 60)
    print("SENTER 3.0 VISION VALIDATION")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent.parent

    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_vision.py",
         "-v", "--tb=short",
         "-x"],  # Stop on first failure
        cwd=project_root
    )

    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("OK - ALL VISION TESTS PASSED")
    else:
        print("FAIL - SOME TESTS FAILED")
    print("=" * 60)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
