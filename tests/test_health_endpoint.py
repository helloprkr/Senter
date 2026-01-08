#!/usr/bin/env python3
"""
Tests for HTTP Health Endpoint (DI-004)
Tests the HTTP server and health response format.
"""

import sys
import time
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_health_monitor_status():
    """Test HealthMonitor status format"""
    from daemon.health_monitor import HealthMonitor

    monitor = HealthMonitor()

    # Register some components
    monitor.components["component_a"] = MagicMock(is_alive=True)
    monitor.components["component_a"].to_dict.return_value = {
        "name": "component_a",
        "is_alive": True
    }

    status = monitor.get_status()

    assert "timestamp" in status
    assert "overall" in status
    assert "components" in status

    return True


def test_health_json_format():
    """Test health response JSON format"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer

    monitor = HealthMonitor()
    server = HealthHTTPServer(monitor, port=18765)

    response, code = server.get_health_response()

    # Check required fields
    assert "status" in response
    assert "uptime_seconds" in response
    assert "components" in response
    assert "summary" in response
    assert "timestamp" in response

    # Check summary fields
    assert "total" in response["summary"]
    assert "alive" in response["summary"]
    assert "dead" in response["summary"]

    return True


def test_health_status_healthy():
    """Test healthy status when all components alive"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer, ComponentHealth

    monitor = HealthMonitor()
    monitor.components["comp_a"] = ComponentHealth("comp_a")
    monitor.components["comp_a"].is_alive = True
    monitor.components["comp_b"] = ComponentHealth("comp_b")
    monitor.components["comp_b"].is_alive = True

    server = HealthHTTPServer(monitor)
    response, code = server.get_health_response()

    assert response["status"] == "healthy"
    assert code == 200
    assert response["summary"]["alive"] == 2

    return True


def test_health_status_degraded():
    """Test degraded status when some components dead"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer, ComponentHealth

    monitor = HealthMonitor()
    monitor.components["comp_a"] = ComponentHealth("comp_a")
    monitor.components["comp_a"].is_alive = True
    monitor.components["comp_b"] = ComponentHealth("comp_b")
    monitor.components["comp_b"].is_alive = False

    server = HealthHTTPServer(monitor)
    response, code = server.get_health_response()

    assert response["status"] == "degraded"
    assert code == 200

    return True


def test_health_status_unhealthy():
    """Test unhealthy status when all components dead"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer, ComponentHealth

    monitor = HealthMonitor()
    monitor.components["comp_a"] = ComponentHealth("comp_a")
    monitor.components["comp_a"].is_alive = False
    monitor.components["comp_b"] = ComponentHealth("comp_b")
    monitor.components["comp_b"].is_alive = False

    server = HealthHTTPServer(monitor)
    response, code = server.get_health_response()

    assert response["status"] == "unhealthy"
    assert code == 503

    return True


def test_health_status_codes():
    """Test correct HTTP status codes"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer, ComponentHealth

    monitor = HealthMonitor()

    # Unknown (no components)
    server = HealthHTTPServer(monitor)
    response, code = server.get_health_response()
    assert code == 200
    assert response["status"] == "unknown"

    # Healthy
    monitor.components["comp"] = ComponentHealth("comp")
    monitor.components["comp"].is_alive = True
    response, code = server.get_health_response()
    assert code == 200

    # Unhealthy
    monitor.components["comp"].is_alive = False
    response, code = server.get_health_response()
    assert code == 503

    return True


def test_health_uptime():
    """Test uptime calculation"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer

    monitor = HealthMonitor()
    server = HealthHTTPServer(monitor)

    # Small delay
    time.sleep(0.1)

    response, _ = server.get_health_response()

    assert response["uptime_seconds"] >= 0.1
    assert response["uptime_seconds"] < 10  # Shouldn't be too long

    return True


def test_health_server_start_stop():
    """Test starting and stopping HTTP server"""
    from daemon.health_monitor import HealthMonitor, HealthHTTPServer

    monitor = HealthMonitor()
    server = HealthHTTPServer(monitor, port=18766)

    # Start server
    server.start()
    assert server._running

    time.sleep(0.2)

    # Stop server
    server.stop()
    assert not server._running

    return True


def test_component_health_to_dict():
    """Test ComponentHealth serialization"""
    from daemon.health_monitor import ComponentHealth

    health = ComponentHealth(
        name="test_comp",
        is_alive=True,
        last_check=time.time(),
        restart_count=2,
        pid=12345
    )

    d = health.to_dict()

    assert d["name"] == "test_comp"
    assert d["is_alive"] is True
    assert d["restart_count"] == 2
    assert d["pid"] == 12345
    assert "last_check" in d

    return True


def test_health_monitor_register():
    """Test component registration"""
    from daemon.health_monitor import HealthMonitor

    monitor = HealthMonitor()

    monitor.register_component("test_comp")

    assert "test_comp" in monitor.components
    assert monitor.components["test_comp"].name == "test_comp"

    return True


def test_health_monitor_report_alive():
    """Test reporting component alive"""
    from daemon.health_monitor import HealthMonitor

    monitor = HealthMonitor()
    monitor.register_component("test_comp")

    before = monitor.components["test_comp"].last_response

    monitor.report_alive("test_comp")

    after = monitor.components["test_comp"].last_response
    assert after > before if before else after > 0
    assert monitor.components["test_comp"].is_alive

    return True


if __name__ == "__main__":
    tests = [
        test_health_monitor_status,
        test_health_json_format,
        test_health_status_healthy,
        test_health_status_degraded,
        test_health_status_unhealthy,
        test_health_status_codes,
        test_health_uptime,
        test_health_server_start_stop,
        test_component_health_to_dict,
        test_health_monitor_register,
        test_health_monitor_report_alive,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
