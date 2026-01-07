#!/usr/bin/env python3
"""
Stress Tests for Senter Daemon

Tests system behavior under load.
Run with: python3 tests/test_stress.py
"""

import sys
import time
import threading
import concurrent.futures
from pathlib import Path

SENTER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SENTER_ROOT))

from daemon.ipc_client import IPCClient


def test_concurrent_queries(num_queries: int = 10) -> dict:
    """Send multiple queries simultaneously"""
    print(f"\nTest: {num_queries} concurrent queries...")

    client = IPCClient()
    results = {"success": 0, "errors": 0, "total_time": 0}

    def send_query(i):
        start = time.time()
        result = client.query(f"What is {i} + {i}?")
        elapsed = time.time() - start
        return {"index": i, "result": result, "time": elapsed}

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_queries) as executor:
        futures = [executor.submit(send_query, i) for i in range(num_queries)]
        query_results = [f.result(timeout=120) for f in futures]

    total_time = time.time() - start_time

    for qr in query_results:
        if "error" not in qr["result"]:
            results["success"] += 1
        else:
            results["errors"] += 1
            print(f"   Error on query {qr['index']}: {qr['result'].get('error')}")

    results["total_time"] = total_time
    results["qps"] = num_queries / total_time if total_time > 0 else 0

    print(f"   Success: {results['success']}/{num_queries}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   QPS: {results['qps']:.2f}")

    return results


def test_rapid_fire_queries(num_queries: int = 20) -> dict:
    """Send queries as fast as possible sequentially"""
    print(f"\nTest: {num_queries} rapid-fire queries...")

    client = IPCClient()
    results = {"success": 0, "errors": 0, "times": []}

    start_time = time.time()

    for i in range(num_queries):
        query_start = time.time()
        result = client.query(f"Reply with just 'ok': {i}")
        query_time = time.time() - query_start

        results["times"].append(query_time)

        if "error" not in result:
            results["success"] += 1
        else:
            results["errors"] += 1

    total_time = time.time() - start_time

    avg_time = sum(results["times"]) / len(results["times"]) if results["times"] else 0
    min_time = min(results["times"]) if results["times"] else 0
    max_time = max(results["times"]) if results["times"] else 0

    print(f"   Success: {results['success']}/{num_queries}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg latency: {avg_time:.2f}s")
    print(f"   Min/Max: {min_time:.2f}s / {max_time:.2f}s")

    return results


def test_large_query() -> dict:
    """Send a very large query"""
    print("\nTest: Large query (10KB)...")

    client = IPCClient()

    # Create 10KB query
    large_text = "test " * 2000  # ~10KB

    start = time.time()
    result = client.query(f"Summarize this in 3 words: {large_text}")
    elapsed = time.time() - start

    success = "error" not in result

    print(f"   Success: {success}")
    print(f"   Time: {elapsed:.2f}s")
    if not success:
        print(f"   Error: {result.get('error')}")

    return {"success": success, "time": elapsed}


def test_status_under_load(duration: int = 10) -> dict:
    """Continuously check status while queries are running"""
    print(f"\nTest: Status under load ({duration}s)...")

    client = IPCClient()
    results = {"status_checks": 0, "status_ok": 0, "queries_sent": 0}
    stop_event = threading.Event()

    def status_checker():
        while not stop_event.is_set():
            result = client.status()
            results["status_checks"] += 1
            if "error" not in result or result.get("error") == "":
                results["status_ok"] += 1
            time.sleep(0.5)

    def query_sender():
        while not stop_event.is_set():
            client.query("Quick test")
            results["queries_sent"] += 1
            time.sleep(0.1)

    # Start threads
    status_thread = threading.Thread(target=status_checker)
    query_thread = threading.Thread(target=query_sender)

    status_thread.start()
    query_thread.start()

    time.sleep(duration)
    stop_event.set()

    status_thread.join()
    query_thread.join()

    print(f"   Status checks: {results['status_checks']} ({results['status_ok']} OK)")
    print(f"   Queries sent: {results['queries_sent']}")

    return results


def test_connection_storm(num_connections: int = 20) -> dict:
    """Create many simultaneous connections"""
    print(f"\nTest: {num_connections} simultaneous connections...")

    def connect_and_check(i):
        client = IPCClient()
        result = client.health()
        return result.get("healthy", False)

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
        futures = [executor.submit(connect_and_check, i) for i in range(num_connections)]
        results = [f.result(timeout=30) for f in futures]

    elapsed = time.time() - start
    success = sum(1 for r in results if r)

    print(f"   Success: {success}/{num_connections}")
    print(f"   Time: {elapsed:.2f}s")

    return {"success": success, "total": num_connections, "time": elapsed}


def run_all_tests():
    """Run all stress tests"""
    print("=" * 50)
    print("SENTER STRESS TESTS")
    print("=" * 50)

    # Check daemon is running
    client = IPCClient()
    if not client.is_daemon_running():
        print("ERROR: Daemon is not running. Start it first.")
        return False

    health = client.health()
    if not health.get("healthy"):
        print("ERROR: Daemon is not healthy")
        return False

    print("\nDaemon is running and healthy. Starting tests...")

    all_passed = True

    # Run tests
    try:
        # Test 1: Concurrent queries
        result = test_concurrent_queries(10)
        if result["errors"] > 2:  # Allow some failures
            all_passed = False

        # Test 2: Rapid fire
        result = test_rapid_fire_queries(20)
        if result["errors"] > 5:
            all_passed = False

        # Test 3: Large query
        result = test_large_query()
        if not result["success"]:
            all_passed = False

        # Test 4: Status under load
        result = test_status_under_load(10)
        if result["status_ok"] < result["status_checks"] * 0.9:
            all_passed = False

        # Test 5: Connection storm
        result = test_connection_storm(20)
        if result["success"] < result["total"] * 0.9:
            all_passed = False

    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL STRESS TESTS PASSED")
    else:
        print("SOME STRESS TESTS FAILED")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
