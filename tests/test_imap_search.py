import unittest
from unittest.mock import patch, MagicMock
import os

# Prevent main() from running CLI logic on import during pytest
os.environ['PYTEST_RUNNING'] = '1'

from main import EmailConnector, Configuration


class MockIMAP:
    def __init__(self, *args, **kwargs):
        self._selected = None

    def login(self, user, pwd):
        return ('OK', [b'Logged in'])

    def select(self, folder):
        self._selected = folder
        return ('OK', [b'1'])

    def search(self, charset, *criteria):
        # emulate combined X-GM-RAW returning IDs for combined tokens
        if 'X-GM-RAW' in criteria:
            query = criteria[1] if len(criteria) > 1 else criteria[0]
            q = query.lower()
            if 'category:promotions' in q and 'category:social' in q:
                return ('OK', [b'101 102 103'])
            if 'category:promotions' in q:
                return ('OK', [b'201 202'])
            if 'category:social' in q:
                return ('OK', [b'301'])
        # fallback generic search
        return ('OK', [b''])

    def fetch(self, eid, _):
        # return a simple RFC822 placeholder
        msg = b"From: test@example.com\r\nSubject: hi\r\nDate: Mon, 01 Jan 2025 00:00:00 +0000\r\n\r\nBody"
        return ('OK', [(b'1 (RFC822 {123}', msg)])

    def close(self):
        return ('OK', [b'Closed'])

    def logout(self):
        return ('BYE', [b'Logged out'])


class TestIMAPSearch(unittest.TestCase):

    @patch('main.imaplib.IMAP4_SSL', new=MockIMAP)
    def test_combined_xgmraw(self):
        cfg = Configuration()
        cfg.config['EMAIL']['username'] = 'a'
        cfg.config['EMAIL']['password'] = 'b'
        connector = EmailConnector(cfg)
        connected = connector.connect()
        self.assertTrue(connected)

        # Set env to use combined
        os.environ['GMAIL_CATEGORIES'] = 'Promotions,Social'
        os.environ['GMAIL_COMBINED'] = '1'

        ids = connector.fetch_email_ids('UNSEEN')
        # combined search should return 3 IDs as per MockIMAP
        self.assertEqual(ids, [b'101', b'102', b'103'])

    @patch('main.imaplib.IMAP4_SSL', new=MockIMAP)
    def test_per_category_fallback(self):
        cfg = Configuration()
        cfg.config['EMAIL']['username'] = 'a'
        cfg.config['EMAIL']['password'] = 'b'
        connector = EmailConnector(cfg)
        connected = connector.connect()
        self.assertTrue(connected)

        # Force per-category searches
        os.environ['GMAIL_CATEGORIES'] = 'Promotions,Social'
        os.environ['GMAIL_COMBINED'] = '0'

        ids = connector.fetch_email_ids('UNSEEN')
        # per-category should return union preserving order: promotions then social
        self.assertEqual(ids, [b'201', b'202', b'301'])


if __name__ == '__main__':
    unittest.main()
