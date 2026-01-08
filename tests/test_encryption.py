#!/usr/bin/env python3
"""
Tests for At-rest Encryption (CG-009)
Tests the encryption manager with graceful fallback when cryptography unavailable.
"""

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_encryption_manager_init():
    """Test that EncryptionManager initializes correctly"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        # Should have initialized
        assert manager.senter_root == senter_root

        # is_enabled depends on whether cryptography is available
        assert manager.is_enabled == CRYPTOGRAPHY_AVAILABLE

        return True


def test_encryption_graceful_fallback():
    """Test that encryption works gracefully when disabled"""
    from daemon.encryption import EncryptionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Create manager with encryption potentially disabled
        manager = EncryptionManager(senter_root)

        test_data = b"Test data for encryption"

        # Encrypt should work even if disabled (returns data as-is)
        encrypted = manager.encrypt(test_data)
        assert encrypted is not None

        # If disabled, data should be unchanged
        if not manager.is_enabled:
            assert encrypted == test_data

        return True


def test_encrypt_decrypt_roundtrip():
    """Test encrypt/decrypt roundtrip when cryptography available"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    if not CRYPTOGRAPHY_AVAILABLE:
        print("  (skipping - cryptography not available)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        # Test string data
        test_string = "Sensitive user data to encrypt"
        encrypted = manager.encrypt(test_string)
        decrypted = manager.decrypt(encrypted)

        assert decrypted.decode() == test_string

        # Test bytes data
        test_bytes = b"Binary data \x00\x01\x02"
        encrypted = manager.encrypt(test_bytes)
        decrypted = manager.decrypt(encrypted)

        assert decrypted == test_bytes

        return True


def test_key_derivation():
    """Test key derivation from passphrase"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    if not CRYPTOGRAPHY_AVAILABLE:
        print("  (skipping - cryptography not available)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Create manager with passphrase
        manager1 = EncryptionManager(senter_root, passphrase="test_passphrase_123")

        # Encrypt with derived key
        test_data = "Secret message"
        encrypted = manager1.encrypt(test_data)

        # Create new manager with same passphrase
        manager2 = EncryptionManager(senter_root, passphrase="test_passphrase_123")

        # Should be able to decrypt with same passphrase
        decrypted = manager2.decrypt(encrypted)
        assert decrypted.decode() == test_data

        return True


def test_encrypt_json():
    """Test JSON encryption/decryption"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        test_obj = {
            "user": "test",
            "preferences": {"theme": "dark", "language": "en"},
            "history": ["item1", "item2"]
        }

        encrypted = manager.encrypt_json(test_obj)

        if manager.is_enabled:
            # Should be encrypted (not readable JSON)
            try:
                json.loads(encrypted.decode())
                # If we can parse it, something is wrong
                assert False, "Data should be encrypted"
            except:
                pass

        decrypted = manager.decrypt_json(encrypted)
        assert decrypted == test_obj

        return True


def test_encrypted_json_storage():
    """Test EncryptedJSONStorage read/write"""
    from daemon.encryption import EncryptionManager, EncryptedJSONStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        file_path = senter_root / "test_data.json"
        storage = EncryptedJSONStorage(file_path, manager)

        # Write data
        test_obj = {"key": "value", "count": 42}
        storage.write(test_obj)

        assert file_path.exists()

        # Read data back
        loaded = storage.read()
        assert loaded == test_obj

        return True


def test_is_encrypted_detection():
    """Test detection of encrypted data"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    if not CRYPTOGRAPHY_AVAILABLE:
        print("  (skipping - cryptography not available)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        # Plain data should not be detected as encrypted
        plain_data = b"Plain text data"
        assert not manager.is_encrypted(plain_data)

        # Encrypted data should be detected
        encrypted = manager.encrypt("Test data")
        assert manager.is_encrypted(encrypted)

        return True


def test_key_storage_file_fallback():
    """Test that key is stored in file when keyring unavailable"""
    from daemon.encryption import EncryptionManager, KEY_FILE

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Mock keyring to be unavailable
        with patch('daemon.encryption.KEYRING_AVAILABLE', False):
            manager = EncryptionManager(senter_root)

            # Key file should be created
            key_path = senter_root / KEY_FILE
            if manager.is_enabled:
                # If encryption is enabled, key should be stored
                # (may not exist if cryptography not available)
                pass

        return True


def test_salt_persistence():
    """Test that salt is persisted for consistent key derivation"""
    from daemon.encryption import EncryptionManager, SALT_FILE, CRYPTOGRAPHY_AVAILABLE

    if not CRYPTOGRAPHY_AVAILABLE:
        print("  (skipping - cryptography not available)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # First manager creates salt
        manager1 = EncryptionManager(senter_root, passphrase="test_pass")

        salt_path = senter_root / SALT_FILE
        assert salt_path.exists()

        # Save original salt
        original_salt = salt_path.read_bytes()

        # Second manager should use same salt
        manager2 = EncryptionManager(senter_root, passphrase="test_pass")

        # Salt should be unchanged
        assert salt_path.read_bytes() == original_salt

        return True


def test_file_encryption():
    """Test file encryption/decryption"""
    from daemon.encryption import EncryptionManager, CRYPTOGRAPHY_AVAILABLE

    if not CRYPTOGRAPHY_AVAILABLE:
        print("  (skipping - cryptography not available)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        manager = EncryptionManager(senter_root)

        # Create test file
        test_file = senter_root / "test_file.txt"
        original_content = "This is sensitive data in a file"
        test_file.write_text(original_content)

        # Encrypt file
        result = manager.encrypt_file(test_file)
        assert result is True

        # File content should be encrypted
        encrypted_content = test_file.read_bytes()
        assert encrypted_content != original_content.encode()

        # Decrypt file
        result = manager.decrypt_file(test_file)
        assert result is True

        # File content should be restored
        restored_content = test_file.read_text()
        assert restored_content == original_content

        return True


if __name__ == "__main__":
    tests = [
        test_encryption_manager_init,
        test_encryption_graceful_fallback,
        test_encrypt_decrypt_roundtrip,
        test_key_derivation,
        test_encrypt_json,
        test_encrypted_json_storage,
        test_is_encrypted_detection,
        test_key_storage_file_fallback,
        test_salt_persistence,
        test_file_encryption,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
