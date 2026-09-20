#!/usr/bin/env python3
"""
Test runner for memoria_ai_anti using Python's built-in unittest module.
Does not require external dependencies.
"""
import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
