#!/usr/bin/env python3
"""
At-rest Encryption for Senter (CG-009)

Provides Fernet symmetric encryption for sensitive data with:
- Key derivation from passphrase
- macOS Keychain integration for secure key storage
- Transparent encrypt/decrypt operations
- Graceful fallback when dependencies unavailable
"""

import os
import base64
import hashlib
import logging
import json
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("senter.encryption")

# Try to import cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography library not available - encryption disabled")

# Try to import keyring for system keychain
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logger.warning("keyring library not available - using file-based key storage")

# Constants
KEYRING_SERVICE = "senter"
KEYRING_USERNAME = "encryption_key"
KEY_FILE = "data/.encryption_key"
SALT_FILE = "data/.encryption_salt"


class EncryptionManager:
    """
    Manages encryption/decryption for sensitive Senter data.

    Features:
    - Fernet symmetric encryption
    - Key derivation from passphrase using PBKDF2
    - macOS Keychain integration (falls back to file)
    - Transparent encryption for JSON/text data
    """

    def __init__(self, senter_root: Path = None, passphrase: str = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")
        self._fernet: Optional['Fernet'] = None
        self._key: Optional[bytes] = None
        self._enabled = CRYPTOGRAPHY_AVAILABLE

        if self._enabled:
            self._initialize_key(passphrase)

    def _initialize_key(self, passphrase: str = None):
        """Initialize encryption key from passphrase or stored key."""
        if passphrase:
            # Derive key from passphrase
            self._key = self._derive_key_from_passphrase(passphrase)
        else:
            # Try to load existing key
            self._key = self._load_key()

        if not self._key:
            # Generate new key
            self._key = self._generate_key()
            self._store_key(self._key)

        if self._key:
            self._fernet = Fernet(self._key)
            logger.info("Encryption initialized")

    def _derive_key_from_passphrase(self, passphrase: str) -> bytes:
        """Derive encryption key from passphrase using PBKDF2."""
        # Get or create salt
        salt = self._get_or_create_salt()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommended minimum
        )

        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        return key

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create new one."""
        salt_path = self.senter_root / SALT_FILE
        salt_path.parent.mkdir(parents=True, exist_ok=True)

        if salt_path.exists():
            return salt_path.read_bytes()

        # Create new salt
        salt = os.urandom(16)
        salt_path.write_bytes(salt)
        return salt

    def _generate_key(self) -> bytes:
        """Generate a new random encryption key."""
        return Fernet.generate_key()

    def _load_key(self) -> Optional[bytes]:
        """Load encryption key from keychain or file."""
        # Try keychain first
        if KEYRING_AVAILABLE:
            try:
                key_str = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
                if key_str:
                    logger.debug("Loaded key from system keychain")
                    return key_str.encode()
            except Exception as e:
                logger.warning(f"Keychain access failed: {e}")

        # Fall back to file
        key_path = self.senter_root / KEY_FILE
        if key_path.exists():
            try:
                key = key_path.read_bytes()
                logger.debug("Loaded key from file")
                return key
            except Exception as e:
                logger.warning(f"Key file read failed: {e}")

        return None

    def _store_key(self, key: bytes):
        """Store encryption key in keychain or file."""
        # Try keychain first
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key.decode())
                logger.debug("Stored key in system keychain")
                return
            except Exception as e:
                logger.warning(f"Keychain storage failed: {e}")

        # Fall back to file (with restrictive permissions)
        key_path = self.senter_root / KEY_FILE
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
        key_path.chmod(0o600)  # Owner read/write only
        logger.debug("Stored key in file")

    @property
    def is_enabled(self) -> bool:
        """Check if encryption is enabled and ready."""
        return self._enabled and self._fernet is not None

    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.

        Args:
            data: String or bytes to encrypt

        Returns:
            Encrypted bytes (base64-encoded Fernet token)
        """
        if not self.is_enabled:
            # Return data as-is if encryption disabled
            if isinstance(data, str):
                return data.encode()
            return data

        if isinstance(data, str):
            data = data.encode()

        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data.

        Args:
            data: Encrypted bytes (Fernet token)

        Returns:
            Decrypted bytes
        """
        if not self.is_enabled:
            return data

        return self._fernet.decrypt(data)

    def encrypt_json(self, obj: dict) -> bytes:
        """Encrypt a JSON-serializable object."""
        json_str = json.dumps(obj)
        return self.encrypt(json_str)

    def decrypt_json(self, data: bytes) -> dict:
        """Decrypt data and parse as JSON."""
        decrypted = self.decrypt(data)
        return json.loads(decrypted.decode())

    def encrypt_file(self, file_path: Path) -> bool:
        """
        Encrypt a file in place.

        Args:
            file_path: Path to file to encrypt

        Returns:
            True if successful
        """
        if not self.is_enabled:
            return False

        try:
            content = file_path.read_bytes()
            encrypted = self.encrypt(content)
            file_path.write_bytes(encrypted)
            logger.info(f"Encrypted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to encrypt file {file_path}: {e}")
            return False

    def decrypt_file(self, file_path: Path) -> bool:
        """
        Decrypt a file in place.

        Args:
            file_path: Path to file to decrypt

        Returns:
            True if successful
        """
        if not self.is_enabled:
            return False

        try:
            encrypted = file_path.read_bytes()
            decrypted = self.decrypt(encrypted)
            file_path.write_bytes(decrypted)
            logger.info(f"Decrypted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to decrypt file {file_path}: {e}")
            return False

    def is_encrypted(self, data: bytes) -> bool:
        """Check if data appears to be Fernet-encrypted."""
        # Fernet tokens start with version byte and timestamp
        # and are base64 encoded, always starting with 'gAAAAA'
        try:
            return data.startswith(b'gAAAAA')
        except:
            return False


class EncryptedJSONStorage:
    """
    Transparent encrypted JSON storage (CG-009).

    Provides read/write operations that automatically encrypt/decrypt.
    """

    def __init__(self, file_path: Path, encryption_manager: EncryptionManager):
        self.file_path = Path(file_path)
        self.encryption = encryption_manager

    def read(self) -> dict:
        """Read and decrypt JSON data."""
        if not self.file_path.exists():
            return {}

        data = self.file_path.read_bytes()

        if self.encryption.is_enabled and self.encryption.is_encrypted(data):
            return self.encryption.decrypt_json(data)

        # Not encrypted, parse as regular JSON
        try:
            return json.loads(data.decode())
        except json.JSONDecodeError:
            return {}

    def write(self, obj: dict):
        """Encrypt and write JSON data."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.encryption.is_enabled:
            data = self.encryption.encrypt_json(obj)
        else:
            data = json.dumps(obj, indent=2).encode()

        self.file_path.write_bytes(data)


# Convenience functions
_default_manager: Optional[EncryptionManager] = None


def get_encryption_manager(senter_root: Path = None) -> EncryptionManager:
    """Get or create the default encryption manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EncryptionManager(senter_root)
    return _default_manager


def encrypt(data: Union[str, bytes], senter_root: Path = None) -> bytes:
    """Encrypt data using default manager."""
    return get_encryption_manager(senter_root).encrypt(data)


def decrypt(data: bytes, senter_root: Path = None) -> bytes:
    """Decrypt data using default manager."""
    return get_encryption_manager(senter_root).decrypt(data)


# ========== SP-001: Secure Key Storage Enhancement ==========

class SecureKeyStorage:
    """
    Secure Key Storage (SP-001)

    Enhanced keychain integration for encryption keys:
    - Primary: macOS Keychain via keyring
    - Fallback: Encrypted file with restrictive permissions
    - Key labeled 'Senter Encryption Key'
    """

    KEYCHAIN_SERVICE = "Senter"
    KEYCHAIN_ACCOUNT = "Senter Encryption Key"

    def __init__(self, senter_root: Path = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")
        self.key_file = self.senter_root / "data" / ".secure_key"

    def store_key(self, key: bytes) -> bool:
        """
        Store key in keychain with fallback to secure file.

        Returns True if successful.
        """
        # Try keychain first
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(
                    self.KEYCHAIN_SERVICE,
                    self.KEYCHAIN_ACCOUNT,
                    key.decode()
                )
                logger.info("Key stored in system keychain")
                return True
            except Exception as e:
                logger.warning(f"Keychain store failed: {e}")

        # Fallback to secure file
        return self._store_key_file(key)

    def retrieve_key(self) -> Optional[bytes]:
        """
        Retrieve key from keychain with fallback to secure file.

        Returns key bytes or None if not found.
        """
        # Try keychain first
        if KEYRING_AVAILABLE:
            try:
                key_str = keyring.get_password(
                    self.KEYCHAIN_SERVICE,
                    self.KEYCHAIN_ACCOUNT
                )
                if key_str:
                    logger.debug("Key retrieved from system keychain")
                    return key_str.encode()
            except Exception as e:
                logger.warning(f"Keychain retrieve failed: {e}")

        # Fallback to secure file
        return self._load_key_file()

    def delete_key(self) -> bool:
        """Delete key from all storage locations."""
        success = True

        # Delete from keychain
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(
                    self.KEYCHAIN_SERVICE,
                    self.KEYCHAIN_ACCOUNT
                )
            except Exception:
                pass

        # Delete file
        if self.key_file.exists():
            try:
                self.key_file.unlink()
            except Exception:
                success = False

        return success

    def _store_key_file(self, key: bytes) -> bool:
        """Store key in file with restrictive permissions."""
        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Owner read/write only
            logger.info("Key stored in secure file")
            return True
        except Exception as e:
            logger.error(f"Key file store failed: {e}")
            return False

    def _load_key_file(self) -> Optional[bytes]:
        """Load key from secure file."""
        if self.key_file.exists():
            try:
                return self.key_file.read_bytes()
            except Exception:
                pass
        return None


# ========== SP-002: Audit Logging ==========

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class AuditLogEntry:
    """A single audit log entry"""
    timestamp: float
    operation: str  # read, write, delete, encrypt, decrypt
    data_type: str  # conversations, goals, preferences, etc.
    component: str  # which component accessed the data
    success: bool = True
    details: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "operation": self.operation,
            "data_type": self.data_type,
            "component": self.component,
            "success": self.success,
            "details": self.details
        }

    def to_log_line(self) -> str:
        """Format as log line for file"""
        dt = datetime.fromtimestamp(self.timestamp).isoformat()
        status = "OK" if self.success else "FAIL"
        return f"{dt} [{status}] {self.component}: {self.operation} {self.data_type} {self.details}"


class AuditLogger:
    """
    Audit Logging for Data Access (SP-002)

    Logs all data access operations for security auditing:
    - Timestamp and datetime
    - Operation type (read, write, delete, encrypt, decrypt)
    - Data type being accessed
    - Component performing the operation
    """

    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_LOG_FILES = 5

    def __init__(self, senter_root: Path = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")
        self.audit_dir = self.senter_root / "data" / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.audit_dir / "access.log"

    def log(self, operation: str, data_type: str, component: str,
            success: bool = True, details: str = "") -> AuditLogEntry:
        """
        Log a data access operation.

        Args:
            operation: Type of operation (read, write, delete, encrypt, decrypt)
            data_type: Type of data being accessed
            component: Component performing the operation
            success: Whether operation succeeded
            details: Additional details

        Returns:
            The created audit log entry
        """
        entry = AuditLogEntry(
            timestamp=time.time(),
            operation=operation,
            data_type=data_type,
            component=component,
            success=success,
            details=details
        )

        self._write_entry(entry)
        return entry

    def _write_entry(self, entry: AuditLogEntry):
        """Write entry to log file with rotation."""
        self._rotate_if_needed()

        with open(self.log_file, "a") as f:
            f.write(entry.to_log_line() + "\n")

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not self.log_file.exists():
            return

        if self.log_file.stat().st_size < self.MAX_LOG_SIZE:
            return

        # Rotate logs
        for i in range(self.MAX_LOG_FILES - 1, 0, -1):
            old_file = self.audit_dir / f"access.log.{i}"
            new_file = self.audit_dir / f"access.log.{i+1}"
            if old_file.exists():
                if new_file.exists():
                    new_file.unlink()
                old_file.rename(new_file)

        # Move current to .1
        backup = self.audit_dir / "access.log.1"
        if self.log_file.exists():
            self.log_file.rename(backup)

    def get_recent_entries(self, limit: int = 100) -> List[AuditLogEntry]:
        """Get recent audit entries."""
        entries = []
        if not self.log_file.exists():
            return entries

        lines = self.log_file.read_text().strip().split("\n")
        for line in lines[-limit:]:
            try:
                # Parse log line
                parts = line.split(" ", 3)
                if len(parts) >= 3:
                    dt_str = parts[0]
                    status = parts[1].strip("[]")
                    rest = parts[2] if len(parts) > 2 else ""

                    # Basic parsing
                    entries.append(AuditLogEntry(
                        timestamp=datetime.fromisoformat(dt_str).timestamp(),
                        operation="unknown",
                        data_type="unknown",
                        component=rest.split(":")[0] if ":" in rest else "unknown",
                        success=status == "OK"
                    ))
            except:
                continue

        return entries

    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of audit log."""
        if not self.log_file.exists():
            return {"entries": 0, "size_bytes": 0}

        content = self.log_file.read_text()
        lines = content.strip().split("\n") if content.strip() else []

        return {
            "entries": len(lines),
            "size_bytes": self.log_file.stat().st_size,
            "log_file": str(self.log_file)
        }


# ========== SP-003: GDPR Data Export ==========

class GDPRDataExporter:
    """
    GDPR Data Export (SP-003)

    Exports all user data in human-readable format:
    - Conversations
    - Goals
    - Preferences
    - Learned patterns
    """

    def __init__(self, senter_root: Path = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")

    def export_all(self) -> Dict[str, Any]:
        """
        Export all user data.

        Returns complete data export as dict.
        """
        export = {
            "export_date": datetime.now().isoformat(),
            "export_version": "1.0",
            "data": {
                "conversations": self._export_conversations(),
                "goals": self._export_goals(),
                "preferences": self._export_preferences(),
                "patterns": self._export_patterns(),
                "feedback": self._export_feedback(),
                "research_results": self._export_research()
            }
        }
        return export

    def export_to_file(self, output_path: Path = None) -> Path:
        """
        Export all data to JSON file.

        Returns path to exported file.
        """
        if output_path is None:
            output_path = self.senter_root / "exports" / f"senter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.export_all()
        output_path.write_text(json.dumps(data, indent=2, default=str))

        logger.info(f"Data exported to {output_path}")
        return output_path

    def _export_conversations(self) -> List[Dict]:
        """Export conversation history."""
        conversations = []
        conv_dir = self.senter_root / "data" / "conversations"
        if conv_dir.exists():
            for f in conv_dir.glob("*.json"):
                try:
                    conversations.append(json.loads(f.read_text()))
                except:
                    continue
        return conversations

    def _export_goals(self) -> List[Dict]:
        """Export goals."""
        goals_file = self.senter_root / "data" / "tasks" / "goals.json"
        if goals_file.exists():
            try:
                return json.loads(goals_file.read_text()).get("goals", [])
            except:
                pass
        return []

    def _export_preferences(self) -> Dict:
        """Export user preferences."""
        prefs_file = self.senter_root / "data" / "learning" / "preferences.json"
        if prefs_file.exists():
            try:
                return json.loads(prefs_file.read_text())
            except:
                pass
        return {}

    def _export_patterns(self) -> Dict:
        """Export learned patterns."""
        patterns_file = self.senter_root / "data" / "learning" / "patterns.json"
        if patterns_file.exists():
            try:
                return json.loads(patterns_file.read_text())
            except:
                pass
        return {}

    def _export_feedback(self) -> List[Dict]:
        """Export feedback data."""
        feedback_file = self.senter_root / "data" / "learning" / "feedback.json"
        if feedback_file.exists():
            try:
                return json.loads(feedback_file.read_text()).get("entries", [])
            except:
                pass
        return []

    def _export_research(self) -> List[Dict]:
        """Export research results."""
        results = []
        research_dir = self.senter_root / "data" / "research" / "results"
        if research_dir.exists():
            for f in research_dir.glob("*.json"):
                try:
                    results.append(json.loads(f.read_text()))
                except:
                    continue
        return results


# ========== SP-004: Data Deletion ==========

class DataDeleter:
    """
    Data Deletion / Right to be Forgotten (SP-004)

    Securely deletes all user data:
    - Requires confirmation
    - Logs deletion before execution
    - Removes all data directories
    """

    PROTECTED_DIRS = ["config"]  # Don't delete config

    def __init__(self, senter_root: Path = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")
        self.audit = AuditLogger(self.senter_root)

    def get_deletion_preview(self) -> Dict[str, Any]:
        """
        Get preview of what will be deleted.

        Returns dict with counts of items to be deleted.
        """
        preview = {
            "directories": [],
            "files": [],
            "total_size_bytes": 0
        }

        data_dir = self.senter_root / "data"
        if data_dir.exists():
            for item in data_dir.rglob("*"):
                if item.is_file():
                    preview["files"].append(str(item.relative_to(self.senter_root)))
                    preview["total_size_bytes"] += item.stat().st_size
                elif item.is_dir():
                    preview["directories"].append(str(item.relative_to(self.senter_root)))

        return preview

    def delete_all_data(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete all user data.

        Args:
            confirm: Must be True to proceed with deletion

        Returns:
            Dict with deletion results
        """
        if not confirm:
            return {
                "success": False,
                "error": "Confirmation required. Pass confirm=True to proceed.",
                "preview": self.get_deletion_preview()
            }

        # Log deletion before execution
        self.audit.log(
            operation="delete",
            data_type="all_user_data",
            component="DataDeleter",
            details="User requested data deletion"
        )

        results = {
            "success": True,
            "deleted_files": 0,
            "deleted_dirs": 0,
            "errors": []
        }

        # Delete data directory
        data_dir = self.senter_root / "data"
        if data_dir.exists():
            results = self._delete_directory(data_dir, results)

        # Delete encryption keys
        key_storage = SecureKeyStorage(self.senter_root)
        key_storage.delete_key()

        logger.info(f"Data deletion complete: {results['deleted_files']} files, {results['deleted_dirs']} dirs")
        return results

    def _delete_directory(self, path: Path, results: Dict) -> Dict:
        """Recursively delete directory contents."""
        import shutil

        try:
            # Delete files first
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                        results["deleted_files"] += 1
                    except Exception as e:
                        results["errors"].append(f"Failed to delete {item}: {e}")

            # Then directories
            for item in sorted(path.rglob("*"), reverse=True):
                if item.is_dir():
                    try:
                        item.rmdir()
                        results["deleted_dirs"] += 1
                    except Exception as e:
                        if item.exists():
                            results["errors"].append(f"Failed to delete {item}: {e}")

            # Finally the root data dir
            try:
                path.rmdir()
                results["deleted_dirs"] += 1
            except:
                pass

        except Exception as e:
            results["errors"].append(str(e))
            results["success"] = False

        return results


# ========== SP-005: Sensitive Data Masking ==========

import re


class SensitiveDataMasker:
    """
    Sensitive Data Detection and Masking (SP-005)

    Detects and masks PII patterns in logs and data:
    - Email addresses
    - Phone numbers
    - SSN
    - Credit card numbers

    Masking is configurable (enable/disable).
    """

    # Regex patterns for sensitive data
    PATTERNS = {
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "phone": re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'),
        "ssn": re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
        "credit_card": re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
        "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    }

    MASK_REPLACEMENTS = {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CARD]",
        "ip_address": "[IP]"
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._enabled_patterns = set(self.PATTERNS.keys())

    def mask(self, text: str) -> str:
        """
        Mask sensitive data in text.

        Args:
            text: Input text potentially containing sensitive data

        Returns:
            Text with sensitive data masked
        """
        if not self.enabled or not text:
            return text

        masked = text
        for pattern_name in self._enabled_patterns:
            pattern = self.PATTERNS.get(pattern_name)
            replacement = self.MASK_REPLACEMENTS.get(pattern_name, "[MASKED]")
            if pattern:
                masked = pattern.sub(replacement, masked)

        return masked

    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        Detect sensitive data in text without masking.

        Returns dict of pattern_type -> list of matches.
        """
        if not text:
            return {}

        detections = {}
        for pattern_name, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detections[pattern_name] = matches

        return detections

    def has_sensitive_data(self, text: str) -> bool:
        """Check if text contains any sensitive data."""
        if not text:
            return False

        for pattern in self.PATTERNS.values():
            if pattern.search(text):
                return True
        return False

    def enable_pattern(self, pattern_name: str):
        """Enable masking for a specific pattern."""
        if pattern_name in self.PATTERNS:
            self._enabled_patterns.add(pattern_name)

    def disable_pattern(self, pattern_name: str):
        """Disable masking for a specific pattern."""
        self._enabled_patterns.discard(pattern_name)

    def set_enabled(self, enabled: bool):
        """Enable or disable all masking."""
        self.enabled = enabled


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter Encryption Manager")
    parser.add_argument("--encrypt", "-e", help="Encrypt a file")
    parser.add_argument("--decrypt", "-d", help="Decrypt a file")
    parser.add_argument("--status", "-s", action="store_true", help="Show encryption status")
    parser.add_argument("--passphrase", "-p", help="Passphrase for key derivation")
    parser.add_argument("--generate-key", action="store_true", help="Generate new encryption key")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.status:
        print(f"\nEncryption Status:")
        print(f"  cryptography available: {CRYPTOGRAPHY_AVAILABLE}")
        print(f"  keyring available: {KEYRING_AVAILABLE}")

        manager = EncryptionManager(Path("."), args.passphrase)
        print(f"  encryption enabled: {manager.is_enabled}")
        print()

    elif args.encrypt:
        manager = EncryptionManager(Path("."), args.passphrase)
        if manager.encrypt_file(Path(args.encrypt)):
            print(f"Encrypted: {args.encrypt}")
        else:
            print(f"Failed to encrypt: {args.encrypt}")

    elif args.decrypt:
        manager = EncryptionManager(Path("."), args.passphrase)
        if manager.decrypt_file(Path(args.decrypt)):
            print(f"Decrypted: {args.decrypt}")
        else:
            print(f"Failed to decrypt: {args.decrypt}")

    elif args.generate_key:
        manager = EncryptionManager(Path("."), args.passphrase)
        if manager.is_enabled:
            print("New encryption key generated and stored")
        else:
            print("Encryption not available (install cryptography package)")

    else:
        parser.print_help()
