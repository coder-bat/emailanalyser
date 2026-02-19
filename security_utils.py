"""
Security utilities for EmailAnalyser
Provides input validation, sanitization, and security helpers.
"""
import re
import html
from typing import Optional


class SecurityUtils:
    """Security utility functions for input validation and sanitization."""
    
    # Pre-compiled regex patterns for performance
    _UID_PATTERN = re.compile(r'^[1-9]\d*$')  # Only positive integers
    _UID_LIST_PATTERN = re.compile(r'^[1-9]\d*(?:,[1-9]\d*)*$')  # Comma-separated positive integers
    _EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    _SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.\-]+$')
    _CSV_INJECTION_CHARS = frozenset(['=', '+', '-', '@', '\t', '\r', '\n'])
    
    # Constants for validation
    MAX_ENV_VALUE_LENGTH = 4096
    MAX_PASSWORD_LENGTH = 256
    MAX_CATEGORIES_LENGTH = 100
    
    @classmethod
    def validate_imap_uid(cls, uid: str) -> bool:
        """
        Validate IMAP UID to prevent command injection.
        Only allows positive integers.
        """
        if not uid:
            return False
        return bool(cls._UID_PATTERN.match(str(uid)))
    
    @classmethod
    def validate_imap_uid_list(cls, uid_list: str) -> bool:
        """
        Validate a comma-separated list of IMAP UIDs.
        """
        if not uid_list:
            return False
        return bool(cls._UID_LIST_PATTERN.match(str(uid_list)))
    
    @classmethod
    def sanitize_for_csv(cls, value: str) -> str:
        """
        Sanitize a string to prevent CSV injection attacks.
        
        CSV injection can occur when cells start with:
        - = (formula)
        - + (formula)
        - - (formula)
        - @ (formula)
        - Tab/CR/LF (command injection)
        
        Prefixes dangerous strings with a single quote to neutralize formulas.
        """
        if not value:
            return value
        
        # Check if value starts with dangerous characters
        if value and value[0] in cls._CSV_INJECTION_CHARS:
            # Prefix with single quote to neutralize formula
            return "'" + value
        
        return value
    
    @classmethod
    def sanitize_html_content(cls, content: str) -> str:
        """
        Escape HTML special characters to prevent XSS.
        """
        if not content:
            return content
        return html.escape(content)
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """
        Validate email address format.
        """
        if not email:
            return False
        return bool(cls._EMAIL_PATTERN.match(email))
    
    @classmethod
    def validate_safe_filename(cls, filename: str) -> bool:
        """
        Validate filename to prevent path traversal.
        Only allows alphanumeric, underscore, hyphen, and dot.
        """
        if not filename:
            return False
        if '..' in filename:
            return False
        if filename.startswith('/'):
            return False
        return bool(cls._SAFE_FILENAME_PATTERN.match(filename))
    
    @classmethod
    def sanitize_env_value(cls, value: Optional[str], max_length: int = None) -> str:
        """
        Sanitize environment variable values to prevent command injection.
        Removes null bytes and control characters.
        
        Args:
            value: The value to sanitize
            max_length: Maximum length (defaults to MAX_ENV_VALUE_LENGTH)
        """
        if value is None:
            return ''
        
        if max_length is None:
            max_length = cls.MAX_ENV_VALUE_LENGTH
        
        value = str(value)
        # Remove null bytes and control characters (except normal whitespace)
        value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
        # Limit length to prevent DoS
        return value[:max_length]
    
    @classmethod
    def validate_categories(cls, categories: str) -> bool:
        """
        Validate categories parameter to prevent injection.
        Only allows alphanumeric, comma, space, and hyphen.
        """
        if not categories:
            return True  # Empty is valid
        return bool(re.match(r'^[a-zA-Z0-9,\s\-]+$', categories))
    
    @classmethod
    def sanitize_password(cls, password: Optional[str]) -> str:
        """
        Sanitize password for environment variable use.
        Limits length and removes null bytes.
        """
        if not password:
            return ''
        # Remove null bytes which could cause issues
        password = password.replace('\x00', '')
        # Limit length
        return password[:cls.MAX_PASSWORD_LENGTH]


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class SecureFilePath:
    """Helper class for secure file path handling."""
    
    @staticmethod
    def validate_path(filepath: str, base_dir: str = None) -> bool:
        """
        Validate that a file path is safe (no path traversal).
        
        Args:
            filepath: The path to validate
            base_dir: Optional base directory that the path must be within
        
        Returns:
            True if path is safe, False otherwise
        """
        if not filepath:
            return False
        
        # Check for path traversal attempts
        if '..' in filepath:
            return False
        
        # Check for absolute paths
        if filepath.startswith('/'):
            return False
        
        # Check for null bytes
        if '\x00' in filepath:
            return False
        
        # If base_dir specified, ensure path is within it
        if base_dir:
            import os
            try:
                full_path = os.path.abspath(os.path.join(base_dir, filepath))
                base_abs = os.path.abspath(base_dir)
                if not full_path.startswith(base_abs):
                    return False
            except (OSError, ValueError):
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename by removing dangerous characters.
        
        Returns:
            Sanitized filename or empty string if invalid
        """
        if not filename:
            return ''
        
        # Remove path separators and null bytes
        filename = filename.replace('/', '').replace('\\', '').replace('\x00', '')
        
        # Remove leading dots (hidden files)
        filename = filename.lstrip('.')
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
