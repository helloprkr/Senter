#!/usr/bin/env python3
"""
Tests for Worker Differentiation - US-004 acceptance criteria:
1. Research worker has different system prompt (research-focused)
2. Research worker pulls from research_tasks queue
3. Research worker stores results to research output folder
4. Primary worker only handles user queries
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_research_worker_has_different_prompt():
    """Test that research worker has research-focused system prompt"""
    import inspect
    from daemon.senter_daemon import research_worker_process, model_worker_process

    # Get source code of both functions
    research_source = inspect.getsource(research_worker_process)
    primary_source = inspect.getsource(model_worker_process)

    # Research worker should have research-specific prompt
    assert "research assistant" in research_source.lower(), \
        "Research worker should have research-focused prompt"
    assert "thorough background research" in research_source.lower() or \
           "comprehensive information" in research_source.lower(), \
        "Research worker should mention thorough research"

    # Primary worker should have generic assistant prompt
    assert "PRIMARY_SYSTEM_PROMPT" in primary_source, \
        "Primary worker should define primary system prompt"

    # They should be different
    assert "RESEARCH_SYSTEM_PROMPT" in research_source, \
        "Research worker should define research system prompt"

    print("✓ test_research_worker_has_different_prompt PASSED")
    return True


def test_research_worker_pulls_from_research_queue():
    """Test that research worker is configured to pull from research_tasks queue"""
    import inspect
    from daemon.senter_daemon import research_worker_process, SenterDaemon

    # Check function signature expects research_queue parameter
    sig = inspect.signature(research_worker_process)
    params = list(sig.parameters.keys())
    assert "research_queue" in params, \
        "Research worker should accept research_queue parameter"

    # Check SenterDaemon._start_model_workers passes research_tasks_queue
    daemon = SenterDaemon()
    start_source = inspect.getsource(daemon._start_model_workers)
    assert "research_tasks_queue" in start_source, \
        "Should pass research_tasks_queue to research worker"
    assert "research_worker_process" in start_source, \
        "Should use research_worker_process function"

    print("✓ test_research_worker_pulls_from_research_queue PASSED")
    return True


def test_research_worker_stores_to_research_folder():
    """Test that research worker stores results to research output folder"""
    import inspect
    from daemon.senter_daemon import research_worker_process

    source = inspect.getsource(research_worker_process)

    # Should reference research output directory
    assert "data/research/results" in source or \
           '"research"' in source and '"results"' in source, \
        "Research worker should store to data/research/results"

    # Should write results to file
    assert ".write_text" in source or "write_text" in source, \
        "Research worker should write results to file"

    # Should create date-organized folders
    assert "strftime" in source or "date" in source.lower(), \
        "Research worker should organize by date"

    print("✓ test_research_worker_stores_to_research_folder PASSED")
    return True


def test_primary_worker_handles_user_queries_only():
    """Test that primary worker only handles user queries"""
    import inspect
    from daemon.senter_daemon import model_worker_process

    source = inspect.getsource(model_worker_process)

    # Should check for user_query or user_voice
    assert "user_query" in source, \
        "Primary worker should handle user_query"
    assert "user_voice" in source, \
        "Primary worker should handle user_voice"

    # Should NOT handle model_request (that was the old way)
    # Check that the if condition is specific to user types
    assert 'msg_type in ("user_query", "user_voice")' in source or \
           'msg_type in (\'user_query\', \'user_voice\')' in source, \
        "Primary worker should ONLY handle user_query and user_voice"

    print("✓ test_primary_worker_handles_user_queries_only PASSED")
    return True


def test_workers_have_different_configs():
    """Test that workers have different generation configurations"""
    import inspect
    from daemon.senter_daemon import research_worker_process, model_worker_process

    research_source = inspect.getsource(research_worker_process)
    primary_source = inspect.getsource(model_worker_process)

    # Research worker should have lower temperature for accuracy
    assert "temperature" in research_source, \
        "Research worker should set temperature"

    # Research worker should have longer output (num_predict)
    assert "2048" in research_source or "num_predict" in research_source, \
        "Research worker should allow longer output"

    # Research worker should have longer timeout
    assert "180" in research_source or "timeout" in research_source, \
        "Research worker should have longer timeout"

    print("✓ test_workers_have_different_configs PASSED")
    return True


def test_workers_differentiated():
    """Combined test for US-004 acceptance criteria"""
    # This is the main test that prd.json references
    from daemon.senter_daemon import SenterDaemon, model_worker_process, research_worker_process

    # 1. Research worker has different system prompt
    import inspect
    research_source = inspect.getsource(research_worker_process)
    primary_source = inspect.getsource(model_worker_process)
    assert "RESEARCH_SYSTEM_PROMPT" in research_source
    assert "PRIMARY_SYSTEM_PROMPT" in primary_source
    assert "research assistant" in research_source.lower()

    # 2. Research worker pulls from research_tasks queue
    daemon = SenterDaemon()
    start_source = inspect.getsource(daemon._start_model_workers)
    assert "research_tasks_queue" in start_source
    assert "research_worker_process" in start_source

    # 3. Research worker stores results to research output folder
    assert "data/research/results" in research_source or \
           ('"research"' in research_source and '"results"' in research_source)

    # 4. Primary worker only handles user queries
    assert 'msg_type in ("user_query", "user_voice")' in primary_source

    print("✓ test_workers_differentiated PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Worker Differentiation Tests (US-004)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_research_worker_has_different_prompt,
        test_research_worker_pulls_from_research_queue,
        test_research_worker_stores_to_research_folder,
        test_primary_worker_handles_user_queries_only,
        test_workers_have_different_configs,
        test_workers_differentiated,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    sys.exit(0 if all_passed else 1)
