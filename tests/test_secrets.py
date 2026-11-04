import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from pybench.secrets import CredentialStore
from server import create_app, utc_now


class CredentialTests(unittest.TestCase):
    def test_encryption_replacement_binding_tampering_and_missing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = sqlite3.connect(root / 'test.sqlite')
            path = root / 'secrets/master.key'
            store = CredentialStore(db, path)
            store.save('first', 'not-a-real-private-api-key')
            db.commit()
            self.assertNotIn(b'not-a-real-private-api-key', (root / 'test.sqlite').read_bytes())
            if os.name != 'nt':
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(CredentialStore(db, path).loaded['first'], 'not-a-real-private-api-key')
            before = db.execute('SELECT nonce FROM endpoint_credentials').fetchone()[0]
            store.save('first', 'replacement')
            self.assertNotEqual(before, db.execute('SELECT nonce FROM endpoint_credentials').fetchone()[0])
            self.assertEqual(CredentialStore(db, path).loaded['first'], 'replacement')
            db.execute("UPDATE endpoint_credentials SET target_id='second'")
            with self.assertRaisesRegex(RuntimeError, '解密失败'):
                CredentialStore(db, path)
            db.execute("UPDATE endpoint_credentials SET target_id='first'")
            original = path.read_bytes()
            path.write_bytes(os.urandom(32))
            with self.assertRaisesRegex(RuntimeError, '解密失败'):
                CredentialStore(db, path)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, '主密钥缺失'):
                CredentialStore(db, path)
            self.assertFalse(path.exists())
            path.write_bytes(original)
            path.chmod(0o600)
            store.delete('first')
            self.assertEqual(CredentialStore(db, path).loaded, {})
            db.close()


class CredentialApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_update_clear_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def client():
                return TestClient(TestServer(create_app(root)))
            web = client()
            await web.start_server()
            owner = next(r.handler.__self__ for r in web.app.router.routes() if r.resource.canonical == '/api/state')
            owner.insert('targets', {'id': 'target', 'name': 'mock', 'base_url': 'http://localhost:9',
                                     'model': 'mock', 'created_at': utc_now()})
            response = await web.post('/api/targets/target/key', json={'api_key': 'secret-original'})
            self.assertEqual(response.status, 200)
            await web.close()
            web = client()
            await web.start_server()
            try:
                owner = next(r.handler.__self__ for r in web.app.router.routes() if r.resource.canonical == '/api/state')
                state = await (await web.get('/api/state')).json()
                self.assertTrue(state['targets'][0]['has_key'])
                self.assertNotIn('secret-original', str(state))
                self.assertEqual(owner.credentials['target'], 'secret-original')
                await web.post('/api/targets/target/key', json={'api_key': 'secret-updated'})
                self.assertEqual(owner.credentials['target'], 'secret-updated')
                await web.delete('/api/targets/target/key')
                self.assertNotIn('target', owner.credentials)
                self.assertEqual(owner.db.execute('SELECT count(*) FROM endpoint_credentials').fetchone()[0], 0)
                await web.post('/api/targets/target/key', json={'api_key': 'secret-final'})
                await web.delete('/api/targets/target')
                self.assertEqual(owner.db.execute('SELECT count(*) FROM endpoint_credentials').fetchone()[0], 0)
            finally:
                await web.close()
