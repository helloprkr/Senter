#!/usr/bin/env python3
"""
Tests for Security & Privacy (SP-001, SP-002, SP-003, SP-004, SP-005)
Tests key storage, audit logging, GDPR export, deletion, and masking.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== SP-001: Secure Key Storage Tests ==========

def test_secure_key_storage_creation():
    """Test SecureKeyStorage can be created (SP-001)"""
    from daemon.encryption import SecureKeyStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SecureKeyStorage(Path(tmpdir))

        assert storage is not None
        assert storage.KEYCHAIN_SERVICE == "Senter"
        assert storage.KEYCHAIN_ACCOUNT == "Senter Encryption Key"

    return True


def test_key_storage_file_fallback():
    """Test key storage file fallback (SP-001)"""
    from daemon.encryption import SecureKeyStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SecureKeyStorage(Path(tmpdir))

        # Store a test key
        test_key = b"test_key_12345678901234567890123"
        result = storage._store_key_file(test_key)

        assert result == True
        assert storage.key_file.exists()

        # Verify permissions (owner read/write only)
        mode = storage.key_file.stat().st_mode
        assert mode & 0o077 == 0  # No group/other permissions

    return True


def test_key_storage_roundtrip():
    """Test key store and retrieve (SP-001)"""
    from daemon.encryption import SecureKeyStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SecureKeyStorage(Path(tmpdir))

        test_key = b"roundtrip_test_key_1234567890123"
        storage._store_key_file(test_key)

        retrieved = storage._load_key_file()

        assert retrieved == test_key

    return True


def test_key_deletion():
    """Test key deletion (SP-001)"""
    from daemon.encryption import SecureKeyStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SecureKeyStorage(Path(tmpdir))

        test_key = b"delete_test_key_12345678901234567"
        storage._store_key_file(test_key)

        assert storage.key_file.exists()

        storage.delete_key()

        assert not storage.key_file.exists()

    return True


# ========== SP-002: Audit Logging Tests ==========

def test_audit_logger_creation():
    """Test AuditLogger can be created (SP-002)"""
    from daemon.encryption import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(Path(tmpdir))

        assert logger is not None
        assert logger.audit_dir.exists()

    return True


def test_audit_log_entry():
    """Test logging an audit entry (SP-002)"""
    from daemon.encryption import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(Path(tmpdir))

        entry = logger.log(
            operation="read",
            data_type="conversations",
            component="test_component",
            success=True,
            details="Test operation"
        )

        assert entry.operation == "read"
        assert entry.data_type == "conversations"
        assert entry.component == "test_component"
        assert entry.success == True

    return True


def test_audit_log_format():
    """Test audit log file format (SP-002)"""
    from daemon.encryption import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(Path(tmpdir))

        logger.log(
            operation="write",
            data_type="goals",
            component="GoalTracker",
            success=True
        )

        assert logger.log_file.exists()

        content = logger.log_file.read_text()
        assert "[OK]" in content
        assert "GoalTracker" in content
        assert "write" in content
        assert "goals" in content

    return True


def test_audit_log_summary():
    """Test audit log summary (SP-002)"""
    from daemon.encryption import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(Path(tmpdir))

        # Log several entries
        for i in range(5):
            logger.log(
                operation="read",
                data_type=f"data_{i}",
                component="test"
            )

        summary = logger.get_log_summary()

        assert summary["entries"] == 5
        assert summary["size_bytes"] > 0

    return True


# ========== SP-003: GDPR Export Tests ==========

def test_gdpr_exporter_creation():
    """Test GDPRDataExporter can be created (SP-003)"""
    from daemon.encryption import GDPRDataExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = GDPRDataExporter(Path(tmpdir))

        assert exporter is not None

    return True


def test_gdpr_export_structure():
    """Test GDPR export structure (SP-003)"""
    from daemon.encryption import GDPRDataExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = GDPRDataExporter(Path(tmpdir))

        export = exporter.export_all()

        assert "export_date" in export
        assert "export_version" in export
        assert "data" in export
        assert "conversations" in export["data"]
        assert "goals" in export["data"]
        assert "preferences" in export["data"]
        assert "patterns" in export["data"]

    return True


def test_gdpr_export_to_file():
    """Test GDPR export to file (SP-003)"""
    from daemon.encryption import GDPRDataExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = GDPRDataExporter(Path(tmpdir))

        output_path = exporter.export_to_file()

        assert output_path.exists()

        # Verify it's valid JSON
        content = json.loads(output_path.read_text())
        assert "export_date" in content

    return True


def test_gdpr_export_completeness():
    """Test GDPR export includes created data (SP-003)"""
    from daemon.encryption import GDPRDataExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test data
        data_dir = Path(tmpdir) / "data" / "learning"
        data_dir.mkdir(parents=True)

        prefs = {"topic_preferences": [{"topic": "python"}]}
        (data_dir / "preferences.json").write_text(json.dumps(prefs))

        exporter = GDPRDataExporter(Path(tmpdir))
        export = exporter.export_all()

        # Should include the preferences we created
        assert export["data"]["preferences"].get("topic_preferences") is not None

    return True


# ========== SP-004: Data Deletion Tests ==========

def test_data_deleter_creation():
    """Test DataDeleter can be created (SP-004)"""
    from daemon.encryption import DataDeleter

    with tempfile.TemporaryDirectory() as tmpdir:
        deleter = DataDeleter(Path(tmpdir))

        assert deleter is not None

    return True


def test_deletion_preview():
    """Test deletion preview (SP-004)"""
    from daemon.encryption import DataDeleter

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test data
        data_dir = Path(tmpdir) / "data" / "learning"
        data_dir.mkdir(parents=True)
        (data_dir / "test.json").write_text("{}")
        (data_dir / "test2.json").write_text("{}")

        deleter = DataDeleter(Path(tmpdir))
        preview = deleter.get_deletion_preview()

        assert len(preview["files"]) >= 2
        assert preview["total_size_bytes"] > 0

    return True


def test_deletion_requires_confirmation():
    """Test deletion requires confirmation (SP-004)"""
    from daemon.encryption import DataDeleter

    with tempfile.TemporaryDirectory() as tmpdir:
        deleter = DataDeleter(Path(tmpdir))

        result = deleter.delete_all_data(confirm=False)

        assert result["success"] == False
        assert "confirm" in result["error"].lower()

    return True


def test_deletion_with_confirmation():
    """Test deletion with confirmation (SP-004)"""
    from daemon.encryption import DataDeleter

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test data
        data_dir = Path(tmpdir) / "data" / "test"
        data_dir.mkdir(parents=True)
        (data_dir / "file.txt").write_text("test")

        deleter = DataDeleter(Path(tmpdir))
        result = deleter.delete_all_data(confirm=True)

        assert result["success"] == True
        assert result["deleted_files"] >= 1

        # Verify data is deleted
        assert not (data_dir / "file.txt").exists()

    return True


# ========== SP-005: Sensitive Data Masking Tests ==========

def test_masker_creation():
    """Test SensitiveDataMasker can be created (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    assert masker is not None
    assert masker.enabled == True
    assert "email" in masker.PATTERNS

    return True


def test_mask_email():
    """Test email masking (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    text = "Contact me at john@example.com for more info"
    masked = masker.mask(text)

    assert "[EMAIL]" in masked
    assert "john@example.com" not in masked

    return True


def test_mask_phone():
    """Test phone masking (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    text = "Call me at 555-123-4567 or 1-800-555-1234"
    masked = masker.mask(text)

    assert "[PHONE]" in masked
    assert "555-123-4567" not in masked

    return True


def test_mask_ssn():
    """Test SSN masking (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    text = "My SSN is 123-45-6789"
    masked = masker.mask(text)

    assert "[SSN]" in masked
    assert "123-45-6789" not in masked

    return True


def test_mask_credit_card():
    """Test credit card masking (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    text = "Card number: 4111-1111-1111-1111"
    masked = masker.mask(text)

    assert "[CARD]" in masked
    assert "4111-1111-1111-1111" not in masked

    return True


def test_detect_sensitive_data():
    """Test sensitive data detection (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    text = "Email: test@example.com, Phone: 555-123-4567"
    detections = masker.detect(text)

    assert "email" in detections
    assert "phone" in detections
    assert "test@example.com" in detections["email"]

    return True


def test_has_sensitive_data():
    """Test has_sensitive_data check (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()

    assert masker.has_sensitive_data("My email is test@example.com") == True
    assert masker.has_sensitive_data("No sensitive data here") == False

    return True


def test_masking_disabled():
    """Test masking can be disabled (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker(enabled=False)

    text = "Email: test@example.com"
    masked = masker.mask(text)

    # Should not mask when disabled
    assert masked == text
    assert "test@example.com" in masked

    return True


def test_disable_specific_pattern():
    """Test disabling specific pattern (SP-005)"""
    from daemon.encryption import SensitiveDataMasker

    masker = SensitiveDataMasker()
    masker.disable_pattern("email")

    text = "Email: test@example.com, Phone: 555-123-4567"
    masked = masker.mask(text)

    # Email should not be masked
    assert "test@example.com" in masked
    # Phone should still be masked
    assert "[PHONE]" in masked

    return True


if __name__ == "__main__":
    tests = [
        # SP-001
        test_secure_key_storage_creation,
        test_key_storage_file_fallback,
        test_key_storage_roundtrip,
        test_key_deletion,
        # SP-002
        test_audit_logger_creation,
        test_audit_log_entry,
        test_audit_log_format,
        test_audit_log_summary,
        # SP-003
        test_gdpr_exporter_creation,
        test_gdpr_export_structure,
        test_gdpr_export_to_file,
        test_gdpr_export_completeness,
        # SP-004
        test_data_deleter_creation,
        test_deletion_preview,
        test_deletion_requires_confirmation,
        test_deletion_with_confirmation,
        # SP-005
        test_masker_creation,
        test_mask_email,
        test_mask_phone,
        test_mask_ssn,
        test_mask_credit_card,
        test_detect_sensitive_data,
        test_has_sensitive_data,
        test_masking_disabled,
        test_disable_specific_pattern,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
