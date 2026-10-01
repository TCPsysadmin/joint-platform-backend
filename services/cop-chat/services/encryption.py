# encryption.py
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AESEncryptor:
    """
    Handles symmetric AES-256-GCM encryption/decryption.
    """

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES key must be 32 bytes (256-bit).")
        self.key = key
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: str) -> dict:
        """
        Encrypt message using AES-GCM.
        """
        nonce = os.urandom(12)  # Required 96-bit nonce for AES-GCM
        ciphertext = self.aesgcm.encrypt(nonce, plaintext.encode(), None)

        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode()
        }

    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> str:
        """
        Decrypt message for debugging/admin-only.
        """
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)

        plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_bytes.decode()
