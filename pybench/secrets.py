"""Authenticated encryption for endpoint credentials; the master key is never stored in SQLite."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CredentialStore:
    def __init__(self, db, key_path: Path):
        self.db = db
        db.execute('CREATE TABLE IF NOT EXISTS endpoint_credentials '
                   '(target_id TEXT PRIMARY KEY, version INTEGER NOT NULL, nonce BLOB NOT NULL, ciphertext BLOB NOT NULL)')
        records = db.execute('SELECT target_id,version,nonce,ciphertext FROM endpoint_credentials').fetchall()
        if not key_path.exists():
            if records:
                raise RuntimeError('API Key 主密钥缺失，请恢复原主密钥；不会生成新密钥覆盖已有凭据')
            key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(os.urandom(32))
                    handle.flush()
                    os.fsync(handle.fileno())
        if key_path.is_symlink() or (os.name != 'nt' and key_path.stat().st_mode & 0o077):
            raise RuntimeError('API Key 主密钥文件必须仅限当前服务账号访问（权限 600）且不能是符号链接')
        key = key_path.read_bytes()
        if len(key) != 32:
            raise RuntimeError('API Key 主密钥格式无效')
        self.cipher = AESGCM(key)
        self.loaded = {}
        for ident, version, nonce, ciphertext in records:
            if version != 1:
                raise RuntimeError('不支持的 API Key 密文版本')
            try:
                self.loaded[ident] = self.cipher.decrypt(nonce, ciphertext, self.aad(ident)).decode('utf-8')
            except (InvalidTag, ValueError, UnicodeError):
                raise RuntimeError('API Key 解密失败，请检查主密钥和数据库；不会覆盖已有凭据') from None

    @staticmethod
    def aad(ident):
        return ('cloudbench:endpoint-key:v1:' + ident).encode('utf-8')

    def save(self, ident, value):
        nonce = os.urandom(12)
        ciphertext = self.cipher.encrypt(nonce, value.encode('utf-8'), self.aad(ident))
        self.db.execute('INSERT OR REPLACE INTO endpoint_credentials VALUES (?,?,?,?)',
                        (ident, 1, nonce, ciphertext))

    def delete(self, ident):
        self.db.execute('DELETE FROM endpoint_credentials WHERE target_id=?', (ident,))
