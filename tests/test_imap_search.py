import unittest
from unittest.mock import patch, MagicMock
import os

# Prevent main() from running CLI logic on import during pytest
os.environ['PYTEST_RUNNING'] = '1'

from main import EmailConnector, Configuration
from security_utils import SecurityUtils, SecureFilePath, ValidationError


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


class TestSecurityUtils(unittest.TestCase):
    """Test security utility functions."""

    def test_validate_imap_uid_valid(self):
        """Test valid IMAP UIDs."""
        self.assertTrue(SecurityUtils.validate_imap_uid('1'))
        self.assertTrue(SecurityUtils.validate_imap_uid('123'))
        self.assertTrue(SecurityUtils.validate_imap_uid('999999'))

    def test_validate_imap_uid_invalid(self):
        """Test invalid IMAP UIDs."""
        self.assertFalse(SecurityUtils.validate_imap_uid(''))
        self.assertFalse(SecurityUtils.validate_imap_uid('0'))
        self.assertFalse(SecurityUtils.validate_imap_uid('01'))
        self.assertFalse(SecurityUtils.validate_imap_uid('abc'))
        self.assertFalse(SecurityUtils.validate_imap_uid('1;2'))
        self.assertFalse(SecurityUtils.validate_imap_uid('1 OR 1'))
        self.assertFalse(SecurityUtils.validate_imap_uid('1\n2'))
        self.assertFalse(SecurityUtils.validate_imap_uid('1\x002'))

    def test_validate_imap_uid_list_valid(self):
        """Test valid IMAP UID lists."""
        self.assertTrue(SecurityUtils.validate_imap_uid_list('1'))
        self.assertTrue(SecurityUtils.validate_imap_uid_list('1,2,3'))
        self.assertTrue(SecurityUtils.validate_imap_uid_list('100,200,300'))

    def test_validate_imap_uid_list_invalid(self):
        """Test invalid IMAP UID lists."""
        self.assertFalse(SecurityUtils.validate_imap_uid_list(''))
        self.assertFalse(SecurityUtils.validate_imap_uid_list('0,1,2'))
        self.assertFalse(SecurityUtils.validate_imap_uid_list('1,abc,3'))
        self.assertFalse(SecurityUtils.validate_imap_uid_list('1;2;3'))
        self.assertFalse(SecurityUtils.validate_imap_uid_list('1 OR 2'))

    def test_sanitize_for_csv(self):
        """Test CSV injection prevention."""
        # Dangerous prefixes should be neutralized
        self.assertEqual(SecurityUtils.sanitize_for_csv('=cmd'), "'=cmd")
        self.assertEqual(SecurityUtils.sanitize_for_csv('+cmd'), "'+cmd")
        self.assertEqual(SecurityUtils.sanitize_for_csv('-cmd'), "'-cmd")
        self.assertEqual(SecurityUtils.sanitize_for_csv('@cmd'), "'@cmd")
        self.assertEqual(SecurityUtils.sanitize_for_csv('\t'), "'\t")
        
        # Safe values should pass through
        self.assertEqual(SecurityUtils.sanitize_for_csv('normal text'), 'normal text')
        self.assertEqual(SecurityUtils.sanitize_for_csv('user@example.com'), 'user@example.com')
        self.assertEqual(SecurityUtils.sanitize_for_csv(''), '')

    def test_sanitize_html_content(self):
        """Test HTML escaping."""
        self.assertEqual(SecurityUtils.sanitize_html_content('<script>'), '&lt;script&gt;')
        self.assertEqual(SecurityUtils.sanitize_html_content('&'), '&amp;')
        self.assertEqual(SecurityUtils.sanitize_html_content('"'), '&quot;')
        self.assertEqual(SecurityUtils.sanitize_html_content("'"), '&#x27;')

    def test_validate_email(self):
        """Test email validation."""
        self.assertTrue(SecurityUtils.validate_email('user@example.com'))
        self.assertTrue(SecurityUtils.validate_email('user.name@example.co.uk'))
        self.assertTrue(SecurityUtils.validate_email('user+tag@example.com'))
        
        self.assertFalse(SecurityUtils.validate_email(''))
        self.assertFalse(SecurityUtils.validate_email('invalid'))
        self.assertFalse(SecurityUtils.validate_email('@example.com'))
        self.assertFalse(SecurityUtils.validate_email('user@'))

    def test_validate_safe_filename(self):
        """Test filename validation."""
        self.assertTrue(SecurityUtils.validate_safe_filename('file.txt'))
        self.assertTrue(SecurityUtils.validate_safe_filename('my-file_name.csv'))
        
        self.assertFalse(SecurityUtils.validate_safe_filename(''))
        self.assertFalse(SecurityUtils.validate_safe_filename('../etc/passwd'))
        self.assertFalse(SecurityUtils.validate_safe_filename('file.txt\x00'))
        self.assertFalse(SecurityUtils.validate_safe_filename('/etc/passwd'))

    def test_sanitize_env_value(self):
        """Test environment variable sanitization."""
        # Control characters should be removed
        self.assertEqual(SecurityUtils.sanitize_env_value('test\x00value'), 'testvalue')
        self.assertEqual(SecurityUtils.sanitize_env_value('test\x01value'), 'testvalue')
        
        # Length should be limited
        long_value = 'a' * 10000
        self.assertEqual(len(SecurityUtils.sanitize_env_value(long_value)), SecurityUtils.MAX_ENV_VALUE_LENGTH)

    def test_validate_categories(self):
        """Test categories validation."""
        self.assertTrue(SecurityUtils.validate_categories('Primary,Social'))
        self.assertTrue(SecurityUtils.validate_categories('Promotions Updates'))
        self.assertTrue(SecurityUtils.validate_categories(''))
        
        self.assertFalse(SecurityUtils.validate_categories('Primary;Social'))
        self.assertFalse(SecurityUtils.validate_categories('Primary\nSocial'))
        self.assertFalse(SecurityUtils.validate_categories('Primary\tSocial'))


class TestSecureFilePath(unittest.TestCase):
    """Test secure file path handling."""

    def test_validate_path_safe(self):
        """Test safe paths."""
        self.assertTrue(SecureFilePath.validate_path('file.txt'))
        self.assertTrue(SecureFilePath.validate_path('subdir/file.csv'))

    def test_validate_path_unsafe(self):
        """Test unsafe paths."""
        self.assertFalse(SecureFilePath.validate_path('../etc/passwd'))
        self.assertFalse(SecureFilePath.validate_path('/etc/passwd'))
        self.assertFalse(SecureFilePath.validate_path('file\x00.txt'))
        self.assertFalse(SecureFilePath.validate_path(''))

    def test_validate_path_with_base_dir(self):
        """Test path validation with base directory."""
        self.assertTrue(SecureFilePath.validate_path('file.txt', '/tmp/output'))
        self.assertFalse(SecureFilePath.validate_path('/other/file.txt', '/tmp/output'))

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        self.assertEqual(SecureFilePath.sanitize_filename('file.txt'), 'file.txt')
        self.assertEqual(SecureFilePath.sanitize_filename('../file.txt'), 'file.txt')
        self.assertEqual(SecureFilePath.sanitize_filename('.hidden'), 'hidden')
        self.assertEqual(SecureFilePath.sanitize_filename(''), '')


class TestEmailConnectorSecurity(unittest.TestCase):
    """Test EmailConnector security features."""

    def setUp(self):
        self.config = Configuration()
        self.config.config['EMAIL']['username'] = 'test@example.com'
        self.config.config['EMAIL']['password'] = 'password'

    @patch('main.imaplib.IMAP4_SSL')
    def test_fetch_rejects_invalid_uid(self, mock_imap):
        """Test that _fetch rejects invalid UIDs."""
        connector = EmailConnector(self.config)
        connector.connection = MagicMock()
        
        # Invalid UIDs should be rejected
        result = connector._fetch('1;2', '(RFC822)')
        self.assertEqual(result, ('NO', [b'']))
        
        result = connector._fetch('abc', '(RFC822)')
        self.assertEqual(result, ('NO', [b'']))
        
        result = connector._fetch('', '(RFC822)')
        self.assertEqual(result, ('NO', [b'']))

    @patch('main.imaplib.IMAP4_SSL')
    def test_fetch_accepts_valid_uid(self, mock_imap):
        """Test that _fetch accepts valid UIDs."""
        connector = EmailConnector(self.config)
        connector.connection = MagicMock()
        connector.connection.uid.return_value = ('OK', [b'1 (RFC822 ...)'])
        
        # Valid UIDs should be accepted
        result = connector._fetch('123', '(RFC822)')
        self.assertNotEqual(result, ('NO', [b'']))
        
        result = connector._fetch('1,2,3', '(RFC822)')
        self.assertNotEqual(result, ('NO', [b'']))


    def test_validate_batch_size(self):
        """Test batch size validation."""
        # Valid values
        is_valid, val = SecurityUtils.validate_batch_size(50)
        self.assertTrue(is_valid)
        self.assertEqual(val, 50)
        
        is_valid, val = SecurityUtils.validate_batch_size(1000)
        self.assertTrue(is_valid)
        self.assertEqual(val, 1000)
        
        # Invalid - too small
        is_valid, val = SecurityUtils.validate_batch_size(0)
        self.assertFalse(is_valid)
        self.assertEqual(val, 100)  # Default
        
        # Invalid - too large
        is_valid, val = SecurityUtils.validate_batch_size(5000)
        self.assertFalse(is_valid)
        self.assertEqual(val, 1000)  # Max
        
        # Invalid - negative
        is_valid, val = SecurityUtils.validate_batch_size(-10)
        self.assertFalse(is_valid)
        self.assertEqual(val, 100)


class TestSearchCriteriaValidation(unittest.TestCase):
    """Test IMAP search criteria validation."""

    def test_valid_search_criteria(self):
        """Test that valid IMAP search criteria are accepted."""
        # These should be valid
        valid_criteria = ['ALL', 'UNSEEN', 'SEEN', 'ANSWERED', 'DELETED', 'FLAGGED']
        for criteria in valid_criteria:
            # The validation in main.py uses a regex pattern
            import re
            valid_criteria_pattern = re.compile(r'^[A-Z0-9\s\(\)\<\>\"\@\[\]\\\+\-\.\:\\]+$')
            self.assertTrue(valid_criteria_pattern.match(criteria.upper()), f"Criteria '{criteria}' should be valid")
        
    def test_invalid_search_criteria(self):
        """Test that invalid search criteria are rejected."""
        # These should be rejected (contain dangerous characters)
        invalid_criteria = [
            'ALL; rm -rf /',  # Command injection attempt
            'UNSEEN | cat /etc/passwd',  # Pipe injection
            'ALL\nDELETE *',  # Newline injection
            'ALL\x00INJECTED',  # Null byte injection
        ]
        import re
        valid_criteria_pattern = re.compile(r'^[A-Z0-9\s\(\)\<\>\"\@\[\]\\\+\-\.\:\\]+$')
        for criteria in invalid_criteria:
            self.assertFalse(valid_criteria_pattern.match(criteria.upper()), f"Criteria '{criteria}' should be invalid")


if __name__ == '__main__':
    unittest.main()
