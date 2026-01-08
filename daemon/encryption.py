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
