
#!/usr/bin/env python3
"""
Email Analysis Tool - A comprehensive solution for analyzing and managing large email inboxes
Author: Email Analytics System
Version: 1.0.0
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import json
import logging
import argparse
import os
import sys
import re
import pickle
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import csv
import getpass
import configparser
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shlex

# Data processing and analysis
import pandas as pd
import numpy as np

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    MultinomialNB = None
    Pipeline = None
    train_test_split = None
    SKLEARN_AVAILABLE = False

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except Exception:
    go = None
    px = None
    make_subplots = None

# Progress bar
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception:
        # fallback: ignore, we'll handle tokenization errors at runtime
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def _normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize datetimes to naive UTC for consistent comparisons."""
    if not dt:
        return dt
    try:
        from datetime import timezone
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return dt

@dataclass
class EmailMessage:
    """Data class representing an email message"""
    uid: str
    subject: str
    sender: str
    sender_email: str
    recipient: str
    date: datetime
    body: str
    headers: Dict[str, str]
    attachments: List[str] = field(default_factory=list)
    category: Optional[str] = None
    importance_score: float = 0.0
    sentiment_score: float = 0.0
    is_read: bool = False
    size: int = 0

class Configuration:
    """Configuration manager for the email analyzer"""
    
    def __init__(self, config_file: str = 'config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_default_config()
        if os.path.exists(config_file):
            self.config.read(config_file)
        else:
            self.save_config()
    
    def load_default_config(self):
        """Load default configuration values"""
        self.config['EMAIL'] = {
            'server': 'imap.gmail.com',
            'port': '993',
            'username': '',
            'password': '',
            'folder': 'INBOX',
            'batch_size': '100',
            'max_emails': '10000',
            'light_fetch': '1',
            'max_body_bytes': '8192',
            'skip_attachment_bodies': '1'
        }
        
        self.config['CATEGORIES'] = {
            'promotional_keywords': 'sale,discount,offer,deal,save,free,limited,exclusive',
            'work_keywords': 'meeting,project,deadline,report,task,assignment,review',
            'personal_keywords': 'family,friend,birthday,congratulations,thank,love',
            'newsletter_keywords': 'newsletter,digest,update,weekly,monthly,subscribe',
            'spam_keywords': 'winner,claim,urgent,act now,million,prize'
        }
        
        self.config['IMPORTANCE'] = {
            'high_priority_senders': '',
            'vip_domains': 'company.com,important.org',
            'urgent_keywords': 'urgent,asap,immediate,critical,emergency',
            'important_keywords': 'important,priority,attention,action required',
            'important_threshold': '0.6'
        }
        
        self.config['ARCHIVING'] = {
            'archive_after_days': '90',
            'archive_low_importance_threshold': '0.3',
            'archive_folder': 'Archive'
        }
        
        self.config['VISUALIZATION'] = {
            'export_format': 'html',
            'chart_style': 'seaborn',
            'color_palette': 'Set2',
            'enabled': '0'
        }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value"""
        try:
            return self.config.get(section, key)
        except:
            return fallback

class EmailConnector:
    """Handles IMAP connection and email fetching"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.connection = None
        self.current_folder = None
    
    def connect(self) -> bool:
        """Establish IMAP connection"""
        try:
            server = self.config.get('EMAIL', 'server')
            port = int(self.config.get('EMAIL', 'port'))
            username = self.config.get('EMAIL', 'username')
            password = self.config.get('EMAIL', 'password')
            
            logger.info(f"Connecting to {server}:{port}")
            self.connection = imaplib.IMAP4_SSL(server, port)
            self.connection.login(username, password)
            logger.info("Successfully connected to email server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            return False

    def _search(self, *args):
        """Try UID search, fallback to SEARCH if UID not supported."""
        try:
            if hasattr(self.connection, 'uid'):
                return self.connection.uid('search', None, *args)
        except Exception:
            pass
        try:
            return self.connection.search(None, *args)
        except Exception:
            return ('NO', [b''])

    def _fetch(self, fetch_set, parts):
        """Try UID fetch, fallback to FETCH if UID not supported."""
        try:
            if hasattr(self.connection, 'uid'):
                return self.connection.uid('fetch', fetch_set, parts)
        except Exception:
            pass
        try:
            return self.connection.fetch(fetch_set, parts)
        except Exception:
            return ('NO', [b''])
    
    def select_folder(self, folder: str = None) -> bool:
        """Select email folder"""
        if not self.connection:
            return False
        
        folder = folder or self.config.get('EMAIL', 'folder', 'INBOX')
        try:
            self.connection.select(folder)
            self.current_folder = folder
            logger.info(f"Selected folder: {folder}")
            return True
        except Exception as e:
            logger.error(f"Failed to select folder {folder}: {str(e)}")
            return False
    
    def fetch_email_ids(self, search_criteria: str = 'ALL') -> List[bytes]:
        """Fetch email IDs based on search criteria"""
        try:
            # Ensure a folder is selected
            folder = self.current_folder or self.config.get('EMAIL', 'folder', 'INBOX')
            try:
                self.connection.select(folder)
            except Exception:
                logger.debug(f"Could not select folder {folder} before search")

            # Check for Gmail category filtering via environment or config
            gmail_cats = os.getenv('GMAIL_CATEGORIES', '')
            server = self.config.get('EMAIL', 'server', '')

            # If gmail categories provided, attempt Gmail-specific searches
            if gmail_cats:
                cats = [c.strip().lower() for c in gmail_cats.split(',') if c.strip()]
                # Special handling: Primary only = Inbox minus Social/Promotions/Updates/Forums
                gm_combined_query = ''
                if len(cats) == 1 and cats[0] == 'primary':
                    primary_strict = str(os.getenv('GMAIL_PRIMARY_STRICT', '0')).lower() in ('1', 'true', 'yes', 'on')
                    if primary_strict:
                        gm_combined_query = 'in:inbox category:primary'
                    else:
                        gm_combined_query = 'in:inbox -category:social -category:promotions -category:updates -category:forums'
                    if search_criteria and search_criteria.upper() != 'ALL':
                        gm_combined_query = f"{gm_combined_query} {search_criteria}"
                    try:
                        typ, data = self._search('X-GM-RAW', gm_combined_query)
                        if data and data[0]:
                            return data[0].split()
                    except Exception as e:
                        logger.debug(f"Primary-only X-GM-RAW search failed: {e}")
                    # fall through to generic handling if that failed

                # map category names to Gmail category tokens
                tokens = []
                for cat in cats:
                    if cat in ('primary', 'promotions', 'social', 'updates', 'forums'):
                        tokens.append(f'category:{cat}')
                    else:
                        tokens.append(cat)

                # Optionally use a combined OR X-GM-RAW query to reduce roundtrips
                use_combined_env = os.getenv('GMAIL_COMBINED', '1')  # default to enabled
                use_combined = str(use_combined_env).lower() not in ('0', 'false', 'no', '')
                if use_combined and len(tokens) > 1:
                    # Build an ORed combined expression: (category:promotions OR category:social)
                    joined = ' OR '.join(tokens)
                    gm_combined_query = f"({joined})"
                    if search_criteria and search_criteria.upper() != 'ALL':
                        gm_combined_query = f"{gm_combined_query} {search_criteria}"

                # Try combined first when requested
                if gm_combined_query:
                    try:
                        typ, data = self._search('X-GM-RAW', gm_combined_query)
                        if data and data[0]:
                            return data[0].split()
                    except Exception as e:
                        logger.debug(f"Combined X-GM-RAW search failed: {e}")

                # Fallback: per-category searches and union results preserving order
                ids_ordered = []
                seen = set()
                for token in tokens:
                    gm_query = token
                    if search_criteria and search_criteria.upper() != 'ALL':
                        gm_query = f"{gm_query} {search_criteria}"

                    try:
                        typ, data = self._search('X-GM-RAW', gm_query)
                    except Exception:
                        # If X-GM-RAW fails (non-Gmail server), fallback to normal search for this token
                        try:
                            if search_criteria and search_criteria.upper() != 'ALL':
                                typ, data = self._search(search_criteria, token)
                            else:
                                typ, data = self._search(token)
                        except Exception as e:
                            logger.debug(f"Search failed for token {token}: {e}")
                            data = [b'']

                    if data and data[0]:
                        for eid in data[0].split():
                            if eid not in seen:
                                seen.add(eid)
                                ids_ordered.append(eid)

                return ids_ordered

            # Default search (no Gmail categories requested) - use UID search
            _, data = self._search(search_criteria)
            if not data or not data[0]:
                return []
            return data[0].split()
        except Exception as e:
            logger.error(f"Failed to fetch email IDs: {str(e)}")
            return []

    def ensure_uids(self, ids: List[bytes]) -> List[bytes]:
        """Resolve a list of IDs (sequence or UID) into UIDs, preserving order.

        Tries non-UID FETCH to map seq->UID; if that yields no matches, tries UID FETCH as identity.
        """
        if not ids:
            return []
        # prepare string list
        id_strs = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in ids]
        mapping = {}
        # Try non-UID fetch to get UID values when id_strs are sequences
        try:
            fetch_set = ','.join(id_strs)
            typ, data = self.connection.fetch(fetch_set, '(UID)')
            if typ == 'OK' and isinstance(data, list):
                for resp in data:
                    if isinstance(resp, tuple) and resp[0]:
                        info = resp[0].decode(errors='ignore') if isinstance(resp[0], bytes) else str(resp[0])
                        m_seq = re.match(r"(\d+)", info)
                        m_uid = re.search(r"UID (\d+)", info)
                        if m_seq and m_uid:
                            mapping[m_seq.group(1)] = m_uid.group(1)
        except Exception:
            pass
        if mapping:
            return [mapping.get(s, s).encode() for s in id_strs]
        # Try UID fetch identity mapping
        try:
            fetch_set = ','.join(id_strs)
            typ, data = self.connection.uid('fetch', fetch_set, '(UID)')
            if typ == 'OK' and isinstance(data, list):
                # If we got responses, assume inputs were UIDs; return as-bytes in same order
                return [s.encode() for s in id_strs]
        except Exception:
            pass
        # Fallback: return as-is
        return [s.encode() for s in id_strs]
    
    def fetch_email(self, email_id: bytes) -> Optional[EmailMessage]:
        """Fetch and parse a single email.

        In light fetch mode, only headers and a limited portion of the text body
        are fetched, along with BODYSTRUCTURE to infer attachment names without
        downloading attachment content.
        """
        try:
            uid_str = email_id.decode() if isinstance(email_id, (bytes, bytearray)) else str(email_id)
            # Read fetch mode settings
            light_fetch = str(self.config.get('EMAIL', 'light_fetch', '1')).lower() in ('1', 'true', 'yes', 'on')
            try:
                max_body_bytes = int(self.config.get('EMAIL', 'max_body_bytes', '8192'))
            except Exception:
                max_body_bytes = 8192

            if light_fetch:
                parts = f'(BODY.PEEK[HEADER] BODY.PEEK[TEXT]<0.{max_body_bytes}> BODYSTRUCTURE FLAGS RFC822.SIZE X-GM-LABELS)'
                typ, data = self._fetch(uid_str, parts)
                if typ != 'OK' or not data:
                    # fallback to full message if light fetch fails
                    typ, data = self._fetch(uid_str, '(RFC822 X-GM-LABELS)')
            else:
                typ, data = self._fetch(uid_str, '(RFC822 X-GM-LABELS)')

            if not data:
                return None

            # Initialize placeholders
            headers_bytes: Optional[bytes] = None
            text_bytes = b''
            structure_info = ''
            flags = []
            gm_labels: List[str] = []
            size_val = 0

            # data can be a list with multiple tuples; iterate and collect
            for resp in data:
                if not isinstance(resp, tuple) or len(resp) < 2:
                    # Some servers include non-tuple markers; skip
                    continue
                info = resp[0].decode(errors='ignore') if isinstance(resp[0], bytes) else str(resp[0])
                payload = resp[1]
                if 'BODY[HEADER' in info:
                    headers_bytes = payload
                elif 'BODY[TEXT' in info:
                    # may be chunked; accumulate
                    if isinstance(payload, bytes):
                        text_bytes += payload
                elif 'BODYSTRUCTURE' in info:
                    structure_info = info
                # Extract FLAGS and SIZE if present
                flags_match = re.search(r'FLAGS \(([^)]*)\)', info)
                if flags_match:
                    flags = [f.strip() for f in flags_match.group(1).split()] if flags_match.group(1).strip() else []
                size_match = re.search(r'RFC822\.SIZE (\d+)', info)
                if size_match:
                    try:
                        size_val = int(size_match.group(1))
                    except Exception:
                        size_val = 0
                # Extract Gmail labels if present
                try:
                    m = re.search(r'X-GM-LABELS \((.*?)\)', info)
                    if m:
                        # Split respecting quotes
                        gm_labels = shlex.split(m.group(1))
                except Exception:
                    pass

            subject = ''
            sender = ''
            sender_email = ''
            recipient = ''
            date = None
            headers: Dict[str, str] = {}
            if headers_bytes:
                try:
                    hdr_msg = email.message_from_bytes(headers_bytes)
                    subject = self._decode_header(hdr_msg.get('Subject', ''))
                    sender = self._decode_header(hdr_msg.get('From', ''))
                    # Extract normalized sender email from common headers
                    sender_email = self._extract_sender_email(hdr_msg)
                    recipient = self._decode_header(hdr_msg.get('To', ''))
                    date_str = hdr_msg.get('Date', '')
                    try:
                        dtmp = parsedate_to_datetime(date_str)
                    except Exception:
                        dtmp = None
                    date = _normalize_datetime(dtmp) if dtmp else None
                    headers = {k: self._decode_header(v) for k, v in hdr_msg.items()}
                except Exception:
                    headers = {}

            body_text = ''
            if text_bytes:
                try:
                    body_text = text_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    body_text = ''

            # Attempt to extract attachment filenames from BODYSTRUCTURE info
            attachments: List[str] = []
            if structure_info:
                try:
                    # Common patterns: NAME "file" or FILENAME "file"
                    # Also handle lower/upper cases
                    for m in re.findall(r'(?i)(?:\bname\b|\bfilename\b)\s+"([^"]+)"', structure_info):
                        decoded = self._decode_header(m)
                        if decoded:
                            attachments.append(decoded)
                except Exception:
                    attachments = []

            # If we fell back to RFC822, parse full message
            if not headers_bytes and data and isinstance(data[0], tuple) and isinstance(data[0][1], (bytes, bytearray)):
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)
                subject = self._decode_header(msg.get('Subject', ''))
                sender = self._decode_header(msg.get('From', ''))
                sender_email = self._extract_sender_email(msg)
                recipient = self._decode_header(msg.get('To', ''))
                date_str = msg.get('Date', '')
                try:
                    dtmp = parsedate_to_datetime(date_str)
                except Exception:
                    dtmp = None
                date = _normalize_datetime(dtmp) if dtmp else None
                body_text = self._extract_body(msg)
                headers = {key: self._decode_header(value) for key, value in msg.items()}
                attachments = self._extract_attachments(msg)
                size_val = len(raw_email)

            # If headers look empty, do a robust fallback to full RFC822 fetch
            try:
                if not (sender or subject):
                    # Try header-only fetch first to reduce payload
                    try:
                        typh, datah = self._fetch(uid_str, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE TO)] FLAGS RFC822.SIZE X-GM-LABELS)')
                        if typh == 'OK' and datah and isinstance(datah, list):
                            for rh in datah:
                                if isinstance(rh, tuple) and isinstance(rh[1], (bytes, bytearray)):
                                    try:
                                        hdr_msg2 = email.message_from_bytes(rh[1])
                                        subj2 = self._decode_header(hdr_msg2.get('Subject', '') or '')
                                        from2 = self._decode_header(hdr_msg2.get('From', '') or '')
                                        if not subject and subj2:
                                            subject = subj2
                                        if not sender and from2:
                                            sender = from2
                                        if not sender_email:
                                            try:
                                                sender_email = self._extract_sender_email(hdr_msg2)
                                            except Exception:
                                                sender_email = ''
                                        if not recipient:
                                            recipient = self._decode_header(hdr_msg2.get('To', '') or '')
                                        if not date:
                                            dstrh = hdr_msg2.get('Date', '')
                                            try:
                                                dtmp = parsedate_to_datetime(dstrh)
                                            except Exception:
                                                dtmp = None
                                            date = _normalize_datetime(dtmp) if dtmp else None
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                if not (sender or subject):
                    typ2, data2 = self._fetch(uid_str, '(RFC822 X-GM-LABELS)')
                    if typ2 == 'OK' and data2 and isinstance(data2, list):
                        for resp2 in data2:
                            if isinstance(resp2, tuple) and isinstance(resp2[1], (bytes, bytearray)):
                                raw2 = resp2[1]
                                try:
                                    msg2 = email.message_from_bytes(raw2)
                                    subj2 = self._decode_header(msg2.get('Subject', '') or '')
                                    from2 = self._decode_header(msg2.get('From', '') or '')
                                    if not subject and subj2:
                                        subject = subj2
                                    if not sender and from2:
                                        sender = from2
                                    if not sender_email:
                                        try:
                                            sender_email = self._extract_sender_email(msg2)
                                        except Exception:
                                            sender_email = ''
                                    if not recipient:
                                        recipient = self._decode_header(msg2.get('To', '') or '')
                                    if not body_text:
                                        body_text = self._extract_body(msg2)
                                    if not headers:
                                        headers = {key: self._decode_header(value) for key, value in msg2.items()}
                                    if not date:
                                        dstr2 = msg2.get('Date', '')
                                        try:
                                            dtmp2 = parsedate_to_datetime(dstr2)
                                        except Exception:
                                            dtmp2 = None
                                        date = _normalize_datetime(dtmp2) if dtmp2 else None
                                    if not size_val:
                                        size_val = len(raw2)
                                except Exception:
                                    pass
            except Exception:
                pass

            # Map Gmail labels to category if available
            preset_category = None
            try:
                lower_labels = [lbl.lower() for lbl in gm_labels]
                if any('category promotions' in lbl for lbl in lower_labels):
                    preset_category = 'promotional'
                elif any('category social' in lbl for lbl in lower_labels):
                    preset_category = 'personal'
                elif any('category updates' in lbl for lbl in lower_labels):
                    preset_category = 'newsletter'
                elif any('category forums' in lbl for lbl in lower_labels):
                    preset_category = 'other'
            except Exception:
                preset_category = None

            # ensure date defaults to now if still missing
            date_final = date if date else _normalize_datetime(datetime.now())
            return EmailMessage(
                uid=uid_str,
                subject=subject,
                sender=sender,
                sender_email=sender_email,
                recipient=recipient,
                date=date_final,
                body=body_text,
                headers=headers,
                attachments=attachments,
                category=preset_category,
                size=size_val
            )
        except Exception as e:
            logger.error(f"Failed to fetch email {email_id}: {str(e)}")
            return None

    def fetch_email_headers(self, uids: List[bytes]) -> Dict[str, Dict[str, Any]]:
        """Batch-fetch headers, flags and size for a list of sequence IDs.

        Returns a mapping uid_str -> {'from':..., 'subject':..., 'date': datetime, 'flags': [...], 'size': int}
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not uids:
            return results

        # chunk by batch size
        try:
            batch_size = int(self.config.get('EMAIL', 'batch_size', '100'))
        except Exception:
            batch_size = 100

        def parse_resp(resp):
            # resp is a tuple like (b'1 (FLAGS (\Seen) RFC822.SIZE 123)', b'headers...')
            try:
                info = resp[0].decode() if isinstance(resp[0], bytes) else str(resp[0])
            except Exception:
                info = ''

            uid_match = re.search(r'UID (\d+)', info)
            # fallback: try to extract sequence number at start
            seq_match = re.match(r"(\d+)", info)
            seq = None
            if uid_match:
                seq = uid_match.group(1)
            elif seq_match:
                seq = seq_match.group(1)

            flags_match = re.search(r'FLAGS \(([^)]*)\)', info)
            flags = []
            if flags_match:
                flags = [f.strip() for f in flags_match.group(1).split()] if flags_match.group(1).strip() else []

            size_match = re.search(r'RFC822\.SIZE (\d+)', info)
            size = int(size_match.group(1)) if size_match else 0

            payload = resp[1] if len(resp) > 1 else b''
            try:
                msg = email.message_from_bytes(payload)
            except Exception:
                msg = None

            subject = ''
            sender = ''
            sender_email = ''
            date_dt = None
            if msg:
                subject = self._decode_header(msg.get('Subject', '') or '')
                sender = self._decode_header(msg.get('From', '') or '')
                # Note: headers batch used mainly for fast-mode; keep email for potential future use
                try:
                    sender_email = self._extract_sender_email(msg)
                except Exception:
                    sender_email = ''
                date_str = msg.get('Date', '')
                try:
                    date_dt = parsedate_to_datetime(date_str)
                except Exception:
                    date_dt = None
                date_dt = _normalize_datetime(date_dt) if date_dt else None

            key = uid_match.group(1) if uid_match else (seq or (payload and getattr(payload, 'get', lambda k, d=None: d)('Message-ID', '')))
            if not key:
                # As a last resort, use a generated index
                key = str(len(results) + 1)

            results[str(key)] = {
                'from': sender,
                'sender_email': sender_email,
                'subject': subject,
                'date': date_dt,
                'flags': flags,
                'size': size
            }

        # Helper to resolve sequence numbers to UIDs
        def resolve_uids(seq_list: List[str]) -> Dict[str, str]:
            mapping = {}
            try:
                fetch_set = ','.join(seq_list)
                typ, data = self.connection.fetch(fetch_set, '(UID)')
                if typ == 'OK' and isinstance(data, list):
                    for resp in data:
                        if isinstance(resp, tuple) and resp[0]:
                            info = resp[0].decode(errors='ignore') if isinstance(resp[0], bytes) else str(resp[0])
                            # pattern: "1 (UID 12345)"
                            m_seq = re.match(r"(\d+)", info)
                            m_uid = re.search(r"UID (\d+)", info)
                            if m_seq and m_uid:
                                mapping[m_seq.group(1)] = m_uid.group(1)
            except Exception:
                return {}
            return mapping

        # operate in chunks
        # Use UID FETCH when we have UIDs; otherwise translate sequence->UID
        uids_strs = [u.decode() if isinstance(u, bytes) else str(u) for u in uids]
        for i in range(0, len(uids_strs), batch_size):
            chunk = uids_strs[i:i+batch_size]
            # try to resolve to UIDs
            seq_to_uid = resolve_uids(chunk)
            fetch_set = ','.join(seq_to_uid.values()) if seq_to_uid else ','.join(chunk)
            try:
                if seq_to_uid:
                    # we have UIDs, use UID FETCH
                    typ, data = self._fetch(fetch_set, '(UID BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS RFC822.SIZE)')
                else:
                    # fallback to non-UID FETCH on sequences
                    typ, data = self.connection.fetch(fetch_set, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS RFC822.SIZE)')
            except Exception as e:
                logger.debug(f"Header fetch failed for {fetch_set}: {e}")
                # try fallback per-id
                for uid in chunk:
                    try:
                        # per-id try UID first, then non-UID
                        typ, data = self._fetch(uid, '(UID BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS RFC822.SIZE)')
                        if typ != 'OK' or not data:
                            typ, data = self.connection.fetch(uid, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS RFC822.SIZE)')
                    except Exception as ex:
                        logger.debug(f"Single header fetch failed for {uid}: {ex}")
                        continue
                    if data:
                        # data may be a list of tuples
                        if isinstance(data, list):
                            for resp in data:
                                if isinstance(resp, tuple):
                                    parse_resp(resp)
                        else:
                            # unexpected format
                            pass
                continue

            if data:
                # data is a list, iterate elements
                for resp in data:
                    if isinstance(resp, tuple):
                        parse_resp(resp)

        # If headers look empty (e.g., all missing From/Subject), do a robust per-id fallback using RFC822.HEADER
        def looks_blank(v: Dict[str, Any]) -> bool:
            return not (v.get('from') or v.get('subject') or v.get('date'))
        if not results or all(looks_blank(v) for v in results.values()):
            logger.debug("Batch header fetch returned blank; falling back to per-ID RFC822.HEADER fetch")
            results = {}
            for uid in uids_strs:
                try:
                    typ, data = self._fetch(uid, '(RFC822.HEADER FLAGS RFC822.SIZE)')
                    if typ != 'OK' or not data:
                        # try non-UID
                        typ, data = self.connection.fetch(uid, '(RFC822.HEADER FLAGS RFC822.SIZE)')
                    if not data:
                        continue
                    for resp in data:
                        if not isinstance(resp, tuple) or len(resp) < 2:
                            continue
                        info = resp[0].decode(errors='ignore') if isinstance(resp[0], bytes) else str(resp[0])
                        payload = resp[1]
                        try:
                            msg = email.message_from_bytes(payload)
                        except Exception:
                            msg = None
                        sender = subject = sender_email = ''
                        date_dt = None
                        if msg:
                            subject = self._decode_header(msg.get('Subject', '') or '')
                            sender = self._decode_header(msg.get('From', '') or '')
                            try:
                                sender_email = self._extract_sender_email(msg)
                            except Exception:
                                sender_email = ''
                            dstr = msg.get('Date', '')
                            try:
                                date_dt = parsedate_to_datetime(dstr)
                            except Exception:
                                date_dt = None
                            date_dt = _normalize_datetime(date_dt) if date_dt else None
                        flags_match = re.search(r'FLAGS \(([^)]*)\)', info)
                        flags = [f.strip() for f in flags_match.group(1).split()] if flags_match and flags_match.group(1).strip() else []
                        size_match = re.search(r'RFC822\.SIZE (\d+)', info)
                        size = int(size_match.group(1)) if size_match else 0
                        results[str(uid)] = {
                            'from': sender,
                            'sender_email': sender_email,
                            'subject': subject,
                            'date': date_dt,
                            'flags': flags,
                            'size': size
                        }
                except Exception as e:
                    logger.debug(f"Per-ID header fallback failed for {uid}: {e}")

        return results
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                try:
                    if encoding:
                        decoded_parts.append(part.decode(encoding))
                    else:
                        decoded_parts.append(part.decode('utf-8', errors='ignore'))
                except:
                    decoded_parts.append(str(part, errors='ignore'))
            else:
                decoded_parts.append(str(part))
        return ' '.join(decoded_parts)
    
    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract email body text"""
        body_parts = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body_parts.append(body)
                    except:
                        pass
                elif content_type == "text/html" and not body_parts:
                    try:
                        html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # Simple HTML to text conversion
                        text = re.sub('<[^<]+?>', '', html_body)
                        body_parts.append(text)
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                body_parts.append(body)
            except:
                pass
        
        return '\n'.join(body_parts)
    
    def _extract_attachments(self, msg: email.message.Message) -> List[str]:
        """Extract attachment filenames"""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append(self._decode_header(filename))
        
        return attachments

    def _extract_sender_email(self, msg: email.message.Message) -> str:
        """Extract and normalize the sender email address from an email message headers.

        Tries 'From', then 'Reply-To', then 'Sender', then 'Return-Path'. Returns lowercase address.
        """
        try:
            from email.utils import parseaddr
        except Exception:
            parseaddr = None
        if not msg:
            return ''
        header_candidates = ['From', 'Reply-To', 'Sender', 'Return-Path']
        for h in header_candidates:
            try:
                raw = msg.get(h, '')
                if not raw:
                    continue
                if parseaddr:
                    _, addr = parseaddr(raw)
                else:
                    # fallback: naive parse
                    m = re.search(r'<([^>]+)>', raw)
                    addr = m.group(1) if m else raw
                addr = (addr or '').strip().strip('<>')
                if '@' in addr:
                    return addr.lower()
            except Exception:
                continue
        return ''
    
    def disconnect(self):
        """Close IMAP connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
                logger.info("Disconnected from email server")
            except:
                pass

class EmailCategorizer:
    """NLP-based email categorization"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.categories = ['promotional', 'work', 'personal', 'newsletter', 'spam', 'other']
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = MultinomialNB()
        self.pipeline = None
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            self.sia = None
    
    def train_classifier(self, emails: List[EmailMessage]):
        """Train the email classifier"""
        # Create training data based on keywords
        training_data = []
        training_labels = []
        
        for email in emails:
            text = f"{email.subject} {email.body}"
            category = self._categorize_by_keywords(text, email.sender)
            training_data.append(text)
            training_labels.append(category)
        
        if len(set(training_labels)) > 1:
            # Create and train pipeline
            self.pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.classifier)
            ])
            
            self.pipeline.fit(training_data, training_labels)
            logger.info("Email classifier trained successfully")
        else:
            logger.warning("Not enough diverse data to train classifier")
    
    def categorize(self, email: EmailMessage) -> str:
        """Categorize an email"""
        if email.category:
            return email.category
        text = f"{email.subject} {email.body}"
        
        # Try ML classification first if available
        if self.pipeline:
            try:
                category = self.pipeline.predict([text])[0]
                email.category = category
                return category
            except:
                pass
        
        # Fallback to keyword-based categorization
        category = self._categorize_by_keywords(text, email.sender)
        email.category = category
        return category
    
    def _categorize_by_keywords(self, text: str, sender: str) -> str:
        """Categorize email based on keywords"""
        text_lower = text.lower()
        
        # Check each category
        categories_scores = {}
        
        for category in ['promotional', 'work', 'personal', 'newsletter', 'spam']:
            keywords = self.config.get('CATEGORIES', f'{category}_keywords', '').split(',')
            score = sum(1 for keyword in keywords if keyword.strip() in text_lower)
            categories_scores[category] = score
        
        # Check sender patterns
        if 'noreply' in sender.lower() or 'newsletter' in sender.lower():
            categories_scores['newsletter'] += 2
        
        # Return category with highest score
        if max(categories_scores.values()) > 0:
            return max(categories_scores, key=categories_scores.get)
        
        return 'other'
    
    def calculate_sentiment(self, email: EmailMessage) -> float:
        """Calculate email sentiment score"""
        text = f"{email.subject} {email.body}"
        if not self.sia:
            email.sentiment_score = 0.0
            return 0.0
        scores = self.sia.polarity_scores(text)
        email.sentiment_score = scores['compound']
        return scores['compound']

class ImportanceScorer:
    """Calculate email importance scores"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.sender_scores = defaultdict(float)
    
    def calculate_importance(self, email: EmailMessage, all_emails: List[EmailMessage] = None) -> float:
        """Calculate importance score for an email"""
        score = 0.0
        
        # 1. Sender importance (30%)
        base_sender = email.sender_email if getattr(email, 'sender_email', '') else email.sender
        sender_score = self._calculate_sender_score(base_sender, all_emails)
        score += sender_score * 0.3
        
        # 2. Subject/content urgency (25%)
        urgency_score = self._calculate_urgency_score(email)
        score += urgency_score * 0.25
        
        # 3. Recency (20%)
        recency_score = self._calculate_recency_score(email.date)
        score += recency_score * 0.2
        
        # 4. Personal relevance (15%)
        relevance_score = self._calculate_relevance_score(email)
        score += relevance_score * 0.15
        
        # 5. Attachments (10%)
        if email.attachments:
            score += 0.1
        
        email.importance_score = min(score, 1.0)
        return email.importance_score
    
    def _calculate_sender_score(self, sender: str, all_emails: List[EmailMessage] = None) -> float:
        """Calculate sender importance score"""
        score = 0.5  # Base score
        
        # Check VIP domains
        vip_domains = self.config.get('IMPORTANCE', 'vip_domains', '').split(',')
        for domain in vip_domains:
            if domain.strip() and domain.strip() in sender.lower():
                score += 0.3
                break
        
        # Check high priority senders
        priority_senders = self.config.get('IMPORTANCE', 'high_priority_senders', '').split(',')
        for priority_sender in priority_senders:
            if priority_sender.strip() and priority_sender.strip() in sender.lower():
                score += 0.5
                break
        
        # Calculate sender frequency if we have email history
        if all_emails and sender in self.sender_scores:
            score += self.sender_scores[sender] * 0.2
        
        return min(score, 1.0)
    
    def _calculate_urgency_score(self, email: EmailMessage) -> float:
        """Calculate urgency score based on keywords"""
        score = 0.0
        text = f"{email.subject} {email.body}".lower()
        
        # Urgent keywords
        urgent_keywords = self.config.get('IMPORTANCE', 'urgent_keywords', '').split(',')
        for keyword in urgent_keywords:
            if keyword.strip() in text:
                score += 0.5
                break
        
        # Important keywords
        important_keywords = self.config.get('IMPORTANCE', 'important_keywords', '').split(',')
        for keyword in important_keywords:
            if keyword.strip() in text:
                score += 0.3
                break
        
        # Check for deadline mentions
        deadline_patterns = [r'deadline', r'due date', r'by \d+', r'before \d+']
        for pattern in deadline_patterns:
            if re.search(pattern, text):
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _calculate_recency_score(self, date: datetime) -> float:
        """Calculate recency score"""
        try:
            dnorm = _normalize_datetime(date)
            days_old = (datetime.utcnow() - dnorm).days
        except Exception:
            days_old = 999
        
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.8
        elif days_old <= 30:
            return 0.5
        elif days_old <= 90:
            return 0.3
        else:
            return 0.1
    
    def _calculate_relevance_score(self, email: EmailMessage) -> float:
        """Calculate personal relevance score"""
        score = 0.5
        
        # Direct recipient (not CC or BCC)
        if email.recipient and '@' in email.recipient:
            if ',' not in email.recipient:  # Single recipient
                score += 0.3
        
        # Work category emails during work hours
        if email.category == 'work':
            hour = email.date.hour
            if 9 <= hour <= 17:  # Work hours
                score += 0.2
        
        return min(score, 1.0)
    
    def update_sender_scores(self, emails: List[EmailMessage]):
        """Update sender frequency scores"""
        sender_counts = Counter((email.sender_email or email.sender) for email in emails)
        total_emails = len(emails)
        
        for sender, count in sender_counts.items():
            self.sender_scores[sender] = count / total_emails

class PatternDetector:
    """Detect patterns in email data"""
    
    def __init__(self):
        self.patterns = {}
    
    def analyze_patterns(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze various patterns in emails"""
        self.patterns = {
            'sender_patterns': self._analyze_sender_patterns(emails),
            'time_patterns': self._analyze_time_patterns(emails),
            'subject_patterns': self._analyze_subject_patterns(emails),
            'category_distribution': self._analyze_category_distribution(emails),
            'attachment_patterns': self._analyze_attachment_patterns(emails),
            'communication_patterns': self._analyze_communication_patterns(emails)
        }
        return self.patterns
    
    def _analyze_sender_patterns(self, emails: List[EmailMessage]) -> Dict:
        """Analyze sender behavior patterns"""
        sender_data = defaultdict(lambda: {'count': 0, 'dates': [], 'categories': [], 'label': ''})
        
        for email in emails:
            key = getattr(email, 'sender_email', '') or email.sender
            ent = sender_data[key]
            ent['count'] += 1
            ent['dates'].append(email.date)
            ent['categories'].append(email.category)
            if not ent['label']:
                ent['label'] = email.sender
        
        # Calculate statistics
        patterns = {
            'top_senders': Counter((getattr(e, 'sender_email', '') or e.sender) for e in emails).most_common(10),
            'sender_categories': {},
            'sender_frequency': {}
        }
        
        for sender, data in sender_data.items():
            if data['count'] >= 5:  # Only analyze frequent senders
                patterns['sender_categories'][sender] = Counter(data['categories']).most_common(1)[0][0]
                
                # Calculate average time between emails
                if len(data['dates']) > 1:
                    sorted_dates = sorted((_normalize_datetime(d) for d in data['dates'] if d))
                    deltas = [(sorted_dates[i+1] - sorted_dates[i]).days 
                             for i in range(len(sorted_dates)-1)]
                    avg_days = sum(deltas) / len(deltas) if deltas else 0
                    patterns['sender_frequency'][sender] = avg_days
        
        return patterns
    
    def _analyze_time_patterns(self, emails: List[EmailMessage]) -> Dict:
        """Analyze time-based patterns"""
        hour_counts = Counter(email.date.hour for email in emails)
        day_counts = Counter(email.date.strftime('%A') for email in emails)
        month_counts = Counter(email.date.month for email in emails)
        
        return {
            'peak_hours': hour_counts.most_common(3),
            'peak_days': day_counts.most_common(3),
            'monthly_distribution': dict(month_counts),
            'hour_distribution': dict(hour_counts),
            'day_distribution': dict(day_counts)
        }
    
    def _analyze_subject_patterns(self, emails: List[EmailMessage]) -> Dict:
        """Analyze subject line patterns"""
        subjects = [email.subject for email in emails if email.subject]
        
        # Common words in subjects
        all_words = []
        stop_words = set(stopwords.words('english'))
        
        for subject in subjects:
            try:
                words = word_tokenize(subject.lower())
            except Exception:
                words = subject.lower().split()
            all_words.extend([w for w in words if w.isalpha() and w not in stop_words])
        
        return {
            'common_words': Counter(all_words).most_common(20),
            'avg_subject_length': sum(len(s) for s in subjects) / len(subjects) if subjects else 0,
            'subjects_with_re': sum(1 for s in subjects if s.lower().startswith('re:')) / len(subjects) if subjects else 0
        }
    
    def _analyze_category_distribution(self, emails: List[EmailMessage]) -> Dict:
        """Analyze category distribution"""
        categories = [email.category for email in emails if email.category]
        return dict(Counter(categories))
    
    def _analyze_attachment_patterns(self, emails: List[EmailMessage]) -> Dict:
        """Analyze attachment patterns"""
        emails_with_attachments = [email for email in emails if email.attachments]
        
        if not emails_with_attachments:
            return {'percentage_with_attachments': 0, 'common_types': []}
        
        # Extract file extensions
        extensions = []
        for email in emails_with_attachments:
            for attachment in email.attachments:
                if '.' in attachment:
                    extensions.append(attachment.split('.')[-1].lower())
        
        return {
            'percentage_with_attachments': len(emails_with_attachments) / len(emails) * 100,
            'common_types': Counter(extensions).most_common(5),
            'avg_attachments_per_email': sum(len(e.attachments) for e in emails_with_attachments) / len(emails_with_attachments)
        }
    
    def _analyze_communication_patterns(self, emails: List[EmailMessage]) -> Dict:
        """Analyze communication patterns"""
        # Response patterns (emails with Re: or Fwd:)
        responses = sum(1 for e in emails if e.subject and ('re:' in e.subject.lower() or 'fwd:' in e.subject.lower()))
        
        # Email length patterns
        lengths = [len(email.body) for email in emails if email.body]
        
        return {
            'response_rate': responses / len(emails) * 100 if emails else 0,
            'avg_email_length': sum(lengths) / len(lengths) if lengths else 0,
            'short_emails': sum(1 for l in lengths if l < 100) / len(lengths) * 100 if lengths else 0,
            'long_emails': sum(1 for l in lengths if l > 1000) / len(lengths) * 100 if lengths else 0
        }

class Visualizer:
    """Create visualizations for email analysis"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.figures = []
        
        # Set style
        style_name = config.get('VISUALIZATION', 'chart_style', 'seaborn')
        try:
            plt.style.use(style_name)
        except Exception:
            try:
                # fall back to seaborn if available
                import seaborn as _sns
                plt.style.use('seaborn')
            except Exception:
                plt.style.use('default')

        palette = config.get('VISUALIZATION', 'color_palette', 'Set2')
        try:
            sns.set_palette(palette)
        except Exception:
            try:
                sns.set_palette('Set2')
            except Exception:
                pass
    
    def create_all_visualizations(self, emails: List[EmailMessage], patterns: Dict[str, Any], output_dir: str = 'visualizations'):
        """Create all visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual visualizations
        self.create_category_distribution(emails, patterns, output_dir)
        self.create_time_analysis(emails, patterns, output_dir)
        self.create_sender_analysis(emails, patterns, output_dir)
        self.create_importance_distribution(emails, output_dir)
        self.create_word_cloud(emails, output_dir)
        self.create_interactive_dashboard(emails, patterns, output_dir)
        
        logger.info(f"All visualizations saved to {output_dir}")
    
    def create_category_distribution(self, emails: List[EmailMessage], patterns: Dict, output_dir: str):
        """Create category distribution pie chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        categories = patterns['category_distribution']
        if categories:
            colors = sns.color_palette('Set2', len(categories))
            wedges, texts, autotexts = ax1.pie(
                categories.values(), 
                labels=categories.keys(), 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title('Email Category Distribution', fontsize=14, fontweight='bold')
            
            # Bar chart
            ax2.bar(categories.keys(), categories.values(), color=colors)
            ax2.set_xlabel('Category', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Email Count by Category', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_time_analysis(self, emails: List[EmailMessage], patterns: Dict, output_dir: str):
        """Create time-based analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Hour distribution
        hour_dist = patterns.get('time_patterns', {}).get('hour_distribution', {})
        hours = list(range(24))
        counts = [hour_dist.get(h, 0) for h in hours]
        axes[0].bar(hours, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].set_xlabel('Hour of Day', fontsize=12)
        axes[0].set_ylabel('Email Count', fontsize=12)
        axes[0].set_title('Email Distribution by Hour', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Day distribution
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_dist = patterns.get('time_patterns', {}).get('day_distribution', {})
        day_counts = [day_dist.get(day, 0) for day in days_order]
        axes[1].bar(range(7), day_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(days_order, rotation=45, ha='right')
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Email Count', fontsize=12)
        axes[1].set_title('Email Distribution by Day', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Monthly trend
        monthly_dist = patterns.get('time_patterns', {}).get('monthly_distribution', {})
        months = list(range(1, 13))
        month_counts = [monthly_dist.get(m, 0) for m in months]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[2].plot(months, month_counts, marker='o', linewidth=2, markersize=8, color='green')
        axes[2].set_xticks(months)
        axes[2].set_xticklabels(month_names)
        axes[2].set_xlabel('Month', fontsize=12)
        axes[2].set_ylabel('Email Count', fontsize=12)
        axes[2].set_title('Monthly Email Trend', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(months, month_counts, alpha=0.3, color='green')

        # Email volume over time
        dates = [e.date.date() for e in emails]
        date_counts = Counter(dates)
        sorted_dates = sorted(date_counts.keys())
        volumes = [date_counts[d] for d in sorted_dates]
        if sorted_dates:
            axes[3].plot(sorted_dates, volumes, linewidth=2, color='steelblue')
            axes[3].fill_between(sorted_dates, volumes, alpha=0.3, color='steelblue')
            axes[3].set_xlabel('Date')
            axes[3].set_ylabel('Number of Emails')
            axes[3].set_title('Email Volume Over Time')
            axes[3].grid(True, alpha=0.3)
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'email_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Analysis chart saved to {filepath}")
    
    def create_word_cloud(self, emails: List[EmailMessage], output_dir: str):
        """Generate word cloud from email subjects"""
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("WordCloud library not installed. Skipping word cloud generation.")
            return

        all_subjects = ' '.join([e.subject for e in emails if e.subject])
        if not all_subjects.strip():
            logger.info("No subject text available for word cloud; skipping.")
            return

        try:
            stop_words = set(['re', 'fwd', 'the', 'to', 'from', 'of', 'and', 'a', 'in', 'is', 'it'])
            wordcloud = WordCloud(width=1600, height=800,
                                  background_color='white',
                                  stopwords=stop_words,
                                  colormap='viridis').generate(all_subjects)
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Email Subjects Word Cloud', fontsize=24, pad=20)
            filepath = os.path.join(output_dir, 'word_cloud.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Word cloud saved to {filepath}")
        except Exception as e:
            logger.debug(f"Skipping word cloud due to error: {e}")

    def create_sender_analysis(self, emails: List[EmailMessage], patterns: Dict, output_dir: str):
        """Create a bar chart for top senders"""
        sender_counts = Counter([e.sender for e in emails])
        top = sender_counts.most_common(10)
        if not top:
            return
        senders, counts = zip(*top)
        plt.figure(figsize=(10, 6))
        y_pos = range(len(senders))
        plt.barh(y_pos, counts, color='teal')
        plt.yticks(y_pos, [s[:40] for s in senders])
        plt.xlabel('Number of Emails')
        plt.title('Top Email Senders')
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'top_senders.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Sender analysis saved to {filepath}")

    def create_importance_distribution(self, emails: List[EmailMessage], output_dir: str):
        """Create importance score histogram"""
        scores = [getattr(e, 'importance_score', 0.0) for e in emails]
        if not scores:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=20, color='coral', edgecolor='black', alpha=0.8)
        plt.xlabel('Importance Score')
        plt.ylabel('Number of Emails')
        plt.title('Distribution of Email Importance Scores')
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'importance_distribution.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Importance distribution saved to {filepath}")

    def create_interactive_dashboard(self, emails: List[EmailMessage], patterns: Dict[str, Any], output_dir: str):
        """Save a lightweight HTML dashboard placeholder using Plotly"""
        try:
            df_dates = Counter([e.date.date().isoformat() for e in emails])
            if not df_dates:
                return
            dates = sorted(df_dates.keys())
            counts = [df_dates[d] for d in dates]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=counts, name='Emails per day'))
            fig.update_layout(title='Email Volume Over Time', xaxis_title='Date', yaxis_title='Count')
            filepath = os.path.join(output_dir, 'dashboard.html')
            fig.write_html(filepath, include_plotlyjs='cdn')
            logger.info(f"Interactive dashboard saved to {filepath}")
        except Exception as e:
            logger.debug(f"Failed to create interactive dashboard: {e}")

class ReportGenerator:
    """Generate comprehensive email analysis reports"""
    
    def __init__(self, output_dir: str = "email_reports", config: Optional[Configuration] = None):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_text_report(self, emails: List[EmailMessage], patterns: Dict[str, Any]) -> str:
        """Generate detailed text report"""
        report = []
        report.append("=" * 80)
        report.append("EMAIL ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total emails analyzed: {len(emails)}")
        
        if emails:
            dates_norm = [_normalize_datetime(e.date) for e in emails if e.date]
            if dates_norm:
                date_range = f"{min(dates_norm).date()} to {max(dates_norm).date()}"
                report.append(f"Date range: {date_range}")
            
            # Category breakdown
            category_counts = Counter([e.category for e in emails])
            report.append("\nCategory Breakdown:")
            for category, count in category_counts.most_common():
                percentage = (count / len(emails)) * 100
                report.append(f"  - {category}: {count} ({percentage:.1f}%)")
            
            # Importance analysis (importance_score is 0-1)
            try:
                imp_thresh = float(self.config.get('IMPORTANCE', 'important_threshold', '0.6')) if self.config else 0.6
            except Exception:
                imp_thresh = 0.6
            important_emails = [e for e in emails if getattr(e, 'importance_score', 0.0) >= imp_thresh]
            report.append(f"\nImportant emails (score >= {imp_thresh}): {len(important_emails)}")
            
            # Top important emails
            if important_emails:
                report.append("\nTop 5 Most Important Emails:")
                for email in sorted(important_emails, key=lambda x: x.importance_score, reverse=True)[:5]:
                    report.append(f"  - [{email.importance_score}] {email.subject[:50]} (from: {email.sender[:30]})")
        
        # Pattern insights
        if patterns:
            report.append("\n" + "=" * 40)
            report.append("PATTERN INSIGHTS")
            report.append("-" * 40)

            if 'time_patterns' in patterns:
                report.append("\nPeak Email Hours:")
                peak_hours = patterns['time_patterns'].get('peak_hours', [])
                for hour, count in peak_hours:
                    report.append(f"  - {hour:02d}:00 - {count} emails")

            if 'sender_patterns' in patterns:
                report.append("\nMost Frequent Senders:")
                top_senders = patterns['sender_patterns'].get('top_senders', [])
                for sender, count in top_senders:
                    report.append(f"  - {sender[:40]}: {count} emails")

            if 'subject_patterns' in patterns:
                report.append("\nCommon Subject Keywords:")
                common = patterns['subject_patterns'].get('common_words', [])
                for word, count in common[:15]:
                    report.append(f"  - '{word}': {count} occurrences")
        
        # Recommendations
        report.append("\n" + "=" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        if emails:
            old_emails = []
            try:
                now_utc_naive = datetime.utcnow()
                for e in emails:
                    d = _normalize_datetime(e.date)
                    if d and (now_utc_naive - d).days > 180:
                        old_emails.append(e)
            except Exception:
                pass
            if old_emails:
                report.append(f"- Consider archiving {len(old_emails)} emails older than 6 months")

            low_importance = [e for e in emails if e.importance_score < 20]
            if low_importance:
                report.append(f"- {len(low_importance)} low-importance emails could be bulk processed")

            newsletter_count = len([e for e in emails if e.category and e.category.lower() == 'newsletter'])
            if newsletter_count > len(emails) * 0.3:
                report.append(f"- High newsletter volume ({newsletter_count}). Consider unsubscribing from unused lists")
        
        report_text = "\n".join(report)
        
        # Save to file
        filepath = os.path.join(self.output_dir, f"email_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Text report saved to {filepath}")
        return report_text
    
    def export_to_csv(self, emails: List[EmailMessage]):
        """Export email data to CSV for further analysis"""
        filepath = os.path.join(self.output_dir, f"email_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['date', 'sender', 'sender_email', 'subject', 'category', 'importance_score', 'has_attachments']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for email in emails:
                writer.writerow({
                    'date': email.date.isoformat(),
                    'sender': email.sender,
                    'sender_email': getattr(email, 'sender_email', ''),
                    'subject': email.subject,
                    'category': email.category,
                    'importance_score': email.importance_score,
                    'has_attachments': bool(email.attachments)
                })
        
        logger.info(f"CSV export saved to {filepath}")

    def export_actionable_senders(self, emails: List[EmailMessage], patterns: Dict[str, Any]):
        """Export actionable sender lists and a summary JSON.

        Returns: (senders_to_delete, important_senders, summary_obj, del_path, imp_path, sum_path)
        """
        # Thresholds from config, else env, else defaults
        if self.config:
            try:
                MIN_COUNT = int(self.config.get('ACTIONS', 'delete_min_count', '5'))
            except Exception:
                MIN_COUNT = 5
            try:
                NEWSLETTER_PCT = float(self.config.get('ACTIONS', 'delete_newsletter_pct', '0.6'))
            except Exception:
                NEWSLETTER_PCT = 0.6
            try:
                AVG_IMPORTANCE_THRESH = float(self.config.get('ACTIONS', 'delete_avg_importance_thresh', '0.15'))
            except Exception:
                AVG_IMPORTANCE_THRESH = 0.15
            try:
                IMPORTANT_AVG_IMPORTANCE = float(self.config.get('ACTIONS', 'important_avg_importance', '0.6'))
            except Exception:
                IMPORTANT_AVG_IMPORTANCE = 0.6
        else:
            try:
                MIN_COUNT = int(os.getenv('ACTION_DELETE_MIN_COUNT', '5'))
            except Exception:
                MIN_COUNT = 5
            try:
                NEWSLETTER_PCT = float(os.getenv('ACTION_DELETE_NEWSLETTER_PCT', '0.6'))
            except Exception:
                NEWSLETTER_PCT = 0.6
            try:
                AVG_IMPORTANCE_THRESH = float(os.getenv('ACTION_DELETE_AVG_IMPORTANCE_THRESH', '0.15'))
            except Exception:
                AVG_IMPORTANCE_THRESH = 0.15
            try:
                IMPORTANT_AVG_IMPORTANCE = float(os.getenv('ACTION_IMPORTANT_AVG_IMPORTANCE', '0.6'))
            except Exception:
                IMPORTANT_AVG_IMPORTANCE = 0.6

        sender_data = defaultdict(list)
        for e in emails:
            key = getattr(e, 'sender_email', '') or e.sender
            sender_data[key].append(e)

        senders_to_delete = []
        important_senders = []
        for sender, msgs in sender_data.items():
            count = len(msgs)
            avg_importance = sum(getattr(m, 'importance_score', 0.0) for m in msgs) / count
            newsletter_pct = sum(1 for m in msgs if m.category and m.category.lower() == 'newsletter') / count
            if count >= MIN_COUNT and (newsletter_pct >= NEWSLETTER_PCT or avg_importance < AVG_IMPORTANCE_THRESH):
                senders_to_delete.append({'sender': sender, 'count': count, 'avg_importance': avg_importance, 'newsletter_pct': newsletter_pct})
            if avg_importance >= IMPORTANT_AVG_IMPORTANCE or any(k in sender.lower() for k in ('ceo', 'hr@', 'boss', 'manager')):
                important_senders.append({'sender': sender, 'count': count, 'avg_importance': avg_importance})

        # Write CSVs
        del_path = os.path.join(self.output_dir, 'senders_to_delete.csv')
        with open(del_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['sender', 'count', 'avg_importance', 'newsletter_pct'])
            writer.writeheader()
            for row in sorted(senders_to_delete, key=lambda r: (-r['count'], r['avg_importance'])):
                writer.writerow(row)

        imp_path = os.path.join(self.output_dir, 'important_senders.csv')
        with open(imp_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['sender', 'count', 'avg_importance'])
            writer.writeheader()
            for row in sorted(important_senders, key=lambda r: (-r['avg_importance'], -r['count'])):
                writer.writerow(row)

        logger.info(f"Senders to delete saved to {del_path}")
        logger.info(f"Important senders saved to {imp_path}")

        summary_obj = {
            'total_senders': len(sender_data),
            'senders_to_delete_count': len(senders_to_delete),
            'important_senders_count': len(important_senders)
        }
        sum_path = os.path.join(self.output_dir, 'summary.json')
        with open(sum_path, 'w', encoding='utf-8') as jf:
            json.dump(summary_obj, jf, indent=2, default=str)
        logger.info(f"Summary JSON saved to {sum_path}")

        return senders_to_delete, important_senders, summary_obj, del_path, imp_path, sum_path

    def export_sender_stats(self, emails: List[EmailMessage], important_threshold: float = 0.6) -> str:
        """Export per-sender statistics: total emails and important emails counts.

        Returns path to the CSV file.
        """
        counts = defaultdict(lambda: {'total': 0, 'important': 0, 'label': ''})
        for e in emails:
            key = getattr(e, 'sender_email', '') or e.sender
            d = counts[key]
            d['total'] += 1
            if getattr(e, 'importance_score', 0.0) >= important_threshold:
                d['important'] += 1
            if not d['label']:
                d['label'] = e.sender
        out_path = os.path.join(self.output_dir, 'sender_stats.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sender_key', 'sender_label', 'total_emails', 'important_emails'])
            for sender, v in sorted(counts.items(), key=lambda kv: (-kv[1]['total'], -kv[1]['important'])):
                writer.writerow([sender, v['label'] or sender, v['total'], v['important']])
        logger.info(f"Sender stats saved to {out_path}")
        return out_path

def main():
    """Main execution function"""
    # Configuration
    config = {
        'email': os.getenv('EMAIL_ADDRESS', '') or Configuration().get('EMAIL', 'username', ''),
        'password': os.getenv('EMAIL_PASSWORD', ''),
        'imap_server': os.getenv('IMAP_SERVER', 'imap.gmail.com'),
        'imap_port': int(os.getenv('IMAP_PORT', '993')),
        'max_emails': int(os.getenv('MAX_EMAILS', '1000')),
        'output_dir': os.getenv('OUTPUT_DIR', 'email_analysis_output')
    }
    
    print("\n" + "=" * 60)
    print("EMAIL ANALYSIS TOOL")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    # Build Configuration object for classes that expect it
    cfg = Configuration()
    # override config values from env
    # Prefer env or loaded config.ini username
    if config['email']:
        cfg.config['EMAIL']['username'] = config['email']
    else:
        cfg.config['EMAIL']['username'] = cfg.get('EMAIL', 'username', '')
    # Prefer env password, then config.ini, otherwise prompt
    pwd = os.getenv('EMAIL_PASSWORD') or config.get('password') or cfg.get('EMAIL', 'password', '')
    if not pwd:
        try:
            pwd = getpass.getpass(prompt=f"Password for {config['email']}: ")
        except Exception:
            pwd = ''
    cfg.config['EMAIL']['password'] = pwd
    cfg.config['EMAIL']['server'] = config['imap_server']
    cfg.config['EMAIL']['port'] = str(config['imap_port'])
    # Optional: override important threshold from environment
    imp_thr_env = os.getenv('IMPORTANT_THRESHOLD')
    if imp_thr_env:
        try:
            _ = float(imp_thr_env)
            if 'IMPORTANCE' not in cfg.config:
                cfg.config['IMPORTANCE'] = {}
            cfg.config['IMPORTANCE']['important_threshold'] = imp_thr_env
        except Exception:
            pass
    # Optional: force full fetch via env
    lf_env = os.getenv('LIGHT_FETCH')
    force_full = os.getenv('FORCE_FULL_FETCH')
    if (lf_env and str(lf_env).lower() in ('0', 'false', 'no')) or (force_full and str(force_full).lower() in ('1','true','yes','on')):
        cfg.config['EMAIL']['light_fetch'] = '0'
    
    try:
        # Initialize components
        print("\n[1/7] Initializing email connector...")
        connector = EmailConnector(cfg)
        # Connect and fetch emails
        print("[2/7] Connecting to email server...")
        t_conn_start = time.time()
        connector.connect()
        t_conn = time.time() - t_conn_start
        logger.info(f"Connection time: {t_conn:.3f}s")

        search_criteria = os.getenv('EMAIL_SEARCH_CRITERIA', 'ALL')
        # Establish a single effective max for the whole run
        try:
            effective_max = int(os.environ.get('MAX_EMAILS', config['max_emails']))
        except Exception:
            effective_max = config['max_emails']
        config['max_emails'] = effective_max

        print(f"[3/7] Fetching up to {effective_max} emails using search '{search_criteria}'...")
        t_search_start = time.time()
        email_ids = connector.fetch_email_ids(search_criteria)
        t_search = time.time() - t_search_start
        logger.info(f"Search returned {len(email_ids)} IDs in {t_search:.3f}s")

        # Batch-fetch headers first to reduce roundtrips
        print("    Performing batched header fetch...")
        t_hdr_start = time.time()
        # Limit to newest N to avoid scanning entire mailbox; newest-first
        if email_ids:
            email_ids_for_headers = list(email_ids[-effective_max:])
            email_ids_for_headers.reverse()
            # Normalize to UIDs to ensure consistent fetch/mapping
            try:
                email_ids_for_headers = connector.ensure_uids(email_ids_for_headers)
            except Exception:
                pass
        else:
            email_ids_for_headers = []

        # Optionally skip header batch to speed up validation runs
        skip_header_batch = str(os.getenv('SKIP_HEADER_BATCH', '0')).lower() in ('1', 'true', 'yes', 'on')
        if skip_header_batch:
            headers_map = {}
            t_hdr = time.time() - t_hdr_start
            logger.info("Skipping header batch (SKIP_HEADER_BATCH=1)")
        else:
            headers_map = connector.fetch_email_headers(email_ids_for_headers)
            t_hdr = time.time() - t_hdr_start
            logger.info(f"Header fetch completed in {t_hdr:.3f}s")

        # If fast mode requested, write a header-only CSV summary and exit early
        fast_mode = os.getenv('FAST_MODE', '0') == '1' or os.getenv('FAST_MODE', '').lower() in ('true', '1')
        if fast_mode:
            hdr_path = os.path.join(config['output_dir'], f"header_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(hdr_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['uid', 'date', 'from', 'subject', 'flags', 'size'])
                # iterate in the UID-normalized, newest-first order and limit to max_emails
                for eid in email_ids_for_headers[:config['max_emails']]:
                    uid_str = eid.decode() if isinstance(eid, bytes) else str(eid)
                    hdr = headers_map.get(uid_str) or {}
                    date_val = hdr.get('date')
                    date_iso = date_val.isoformat() if isinstance(date_val, datetime) else (str(date_val) if date_val else '')
                    from_val = hdr.get('from', '')
                    subject_val = hdr.get('subject', '')
                    flags_val = ','.join(hdr.get('flags', [])) if hdr.get('flags') else ''
                    size_val = hdr.get('size', 0)
                    writer.writerow([uid_str, date_iso, from_val, subject_val, flags_val, size_val])

            logger.info(f"Header-only summary saved to {hdr_path}")
            print(f"Header-only summary saved to: {hdr_path}")
            # early return to avoid expensive body fetch & analysis
            return

    # Decide which UIDs need full-body fetch
        uids_to_fetch = []
        # criteria: unread (UNSEEN) or default - fetch all until max_emails
        want_unread = (search_criteria.upper() == 'UNSEEN')

        # Map email_ids order to ensure consistent ordering
        for i, eid in enumerate(email_ids_for_headers):
            if len(uids_to_fetch) >= effective_max:
                break
            uid_str = eid.decode() if isinstance(eid, bytes) else str(eid)
            hdr = headers_map.get(uid_str) or headers_map.get(str(i+1))
            # If we fetched headers but cannot find mapping, include UID
            if skip_header_batch or not hdr:
                uids_to_fetch.append(eid)
                continue

            flags = hdr.get('flags', [])
            is_unread = '\\Seen' not in flags and '\\Answered' not in flags
            if want_unread:
                if is_unread:
                    uids_to_fetch.append(eid)
            else:
                # not specifically unread: still fetch but limited by max_emails
                uids_to_fetch.append(eid)

        print(f"    Will fetch full bodies for {len(uids_to_fetch)} messages (up to {effective_max})")
        t_fetch_start = time.time()

        emails = []
        # Fetch bodies with optional parallelism. IMAP connection is not thread-safe; default to serial.
        try:
            batch_size_val = int(connector.config.get('EMAIL', 'batch_size', '100'))
        except Exception:
            batch_size_val = 100
        # Determine workers from env, default 1 (serial)
        try:
            env_workers = int(os.getenv('FETCH_WORKERS', '1'))
        except Exception:
            env_workers = 1
        max_workers = max(1, min(env_workers, 8))
        if max_workers > 1:
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(connector.fetch_email, uid) for uid in uids_to_fetch[:effective_max]]
                    for f in as_completed(futures):
                        em = f.result()
                        if em:
                            emails.append(em)
            except Exception:
                # fallback to serial fetch
                for i, eid in enumerate(uids_to_fetch[:effective_max]):
                    em = connector.fetch_email(eid)
                    if em:
                        emails.append(em)
        else:
            # Serial fetch
            for i, eid in enumerate(uids_to_fetch[:effective_max]):
                em = connector.fetch_email(eid)
                if em:
                    emails.append(em)

        t_fetch = time.time() - t_fetch_start
        logger.info(f"Body fetch completed in {t_fetch:.3f}s")
        print(f"    Retrieved {len(emails)} emails")
        
        if not emails:
            print("\nNo emails found to analyze.")
            return
        
        # Process emails
        print("[4/7] Processing and categorizing emails...")
        categorizer = EmailCategorizer(cfg)
        categorized_emails = []
        for e in emails:
            categorizer.categorize(e)
            categorized_emails.append(e)
        # Calculate importance
        scorer = ImportanceScorer(cfg)
        scorer.update_sender_scores(categorized_emails)
        for e in categorized_emails:
            scorer.calculate_importance(e, categorized_emails)
        
        # Detect patterns
        print("[5/7] Detecting patterns...")
        pattern_detector = PatternDetector()
        patterns = pattern_detector.analyze_patterns(categorized_emails)
        
        # Generate visualizations (disabled by default)
        vis_enabled = os.getenv('VIS_ENABLED', cfg.get('VISUALIZATION', 'enabled', '0'))
        if str(vis_enabled).lower() in ('1', 'true', 'yes', 'on'):
            print("[6/7] Creating visualizations...")
            visualizer = Visualizer(cfg)
            patterns = pattern_detector.analyze_patterns(categorized_emails)
            visualizer.create_all_visualizations(categorized_emails, patterns, output_dir=config['output_dir'])
        else:
            print("[6/7] Skipping visualizations (disabled)")

        # Generate report and data exports
        print("[7/7] Generating reports...")
        report_gen = ReportGenerator(config['output_dir'], cfg)
        report_text = report_gen.generate_text_report(categorized_emails, patterns)
        report_gen.export_to_csv(categorized_emails)
        try:
            imp_thresh = float(cfg.get('IMPORTANCE', 'important_threshold', '0.6'))
        except Exception:
            imp_thresh = 0.6
        sender_stats_path = report_gen.export_sender_stats(categorized_emails, important_threshold=imp_thresh)
        # Export actionable outputs and print summaries
        try:
            senders_to_delete, important_senders, summary_obj, del_path, imp_path, sum_path = report_gen.export_actionable_senders(categorized_emails, patterns)
            if senders_to_delete:
                print("\nTop senders to consider deleting/unsubscribing:")
                for row in sorted(senders_to_delete, key=lambda r: (-r['count'], r['avg_importance']))[:10]:
                    print(f"  - {row['sender'][:60]:60} | count={row['count']:3} | avg_imp={row['avg_importance']:.2f} | news_pct={row['newsletter_pct']:.2f}")
            else:
                print("\nNo strong candidates found for deletion based on current heuristics.")

            if important_senders:
                print("\nTop important senders:")
                for row in sorted(important_senders, key=lambda r: (-r['avg_importance'], -r['count']))[:10]:
                    print(f"  - {row['sender'][:60]:60} | count={row['count']:3} | avg_imp={row['avg_importance']:.2f}")
            else:
                print("\nNo senders marked as important by the heuristics.")

            # Always confirm actionable counts
            print(f"\nActionable counts: senders_to_delete={len(senders_to_delete)}, important_senders={len(important_senders)}")
            print(f"Summary JSON: {sum_path}")
        except Exception as e:
            logger.error(f"Failed to export actionable senders: {e}")

        # If fast mode requested, also write a header-only CSV and small summary
        fast_mode = os.getenv('FAST_MODE', '0') == '1' or os.getenv('FAST_MODE', '').lower() in ('true', '1')
        if fast_mode:
            hdr_path = os.path.join(config['output_dir'], 'header_summary.csv')
            with open(hdr_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['uid', 'date', 'from', 'subject', 'flags', 'size'])
                # headers_map keys may be sequence numbers; write values for the uids we fetched
                for e in emails:
                    writer.writerow([e.uid, e.date.isoformat() if e.date else '', e.sender, e.subject, ','.join(e.headers.get('Flags', [])) if e.headers else '', e.size])
            logger.info(f"Header summary saved to {hdr_path}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {config['output_dir']}/")
        print("\nGenerated files:")
        print("  - email_report_*.txt (detailed text report)")
        print("  - email_data_*.csv (raw data export)")
        print("  - sender_stats.csv (per-sender totals and important counts)")
        print("  - senders_to_delete.csv (suggested senders to consider unsubscribing/deleting)")
        print("  - important_senders.csv (senders marked as important)")
        print("  - summary.json (quick actionable summary)")
        
        # Show quick stats
        print(f"\nQuick Statistics:")
        print(f"  - Total emails: {len(categorized_emails)}")
        print(f"  - Important emails: {len([e for e in categorized_emails if getattr(e, 'importance_score', 0.0) >= imp_thresh])}")
        print(f"  - Categories found: {len(set(e.category for e in categorized_emails))}")
        # Print top senders summary
        sc = Counter((getattr(e, 'sender_email', '') or e.sender) for e in categorized_emails)
        print("  - Top senders:")
        label_map = {}
        for e in categorized_emails:
            key = getattr(e, 'sender_email', '') or e.sender
            if key not in label_map:
                label_map[key] = e.sender
        for sender_key, cnt in sc.most_common(10):
            imp = sum(1 for e in categorized_emails if (getattr(e, 'sender_email', '') or e.sender) == sender_key and getattr(e, 'importance_score', 0.0) >= imp_thresh)
            label = label_map.get(sender_key, sender_key)
            print(f"      {label[:60]:60} | total={cnt:4} | important={imp:4}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Check your email credentials")
        print("  2. For Gmail, use an app-specific password")
        print("  3. Ensure IMAP is enabled in your email settings")
        print("  4. Check your internet connection")
    
    finally:
        try:
            connector.disconnect()
        except:
            pass

if __name__ == "__main__" and not os.getenv('PYTEST_RUNNING'):
    # Setup argument parser for command-line usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and organize your email inbox')
    parser.add_argument('--email', help='Email address', default=None)
    parser.add_argument('--server', help='IMAP server address', default=None)
    parser.add_argument('--port', type=int, help='IMAP port', default=None)
    parser.add_argument('--max-emails', type=int, default=None, help='Maximum emails to analyze')
    parser.add_argument('--unread', action='store_true', help='Only analyze unread (UNSEEN) emails')
    parser.add_argument('--gmail-categories', help='Comma-separated Gmail categories to search (Primary,Social,Promotions,Updates)', default=None)
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--fast-mode', action='store_true', help='Header-only fetch; write small header report and skip full body processing')
    
    args = parser.parse_args()
    
    # Override config with command-line arguments if provided
    if args.email is not None:
        os.environ['EMAIL_ADDRESS'] = args.email
    if args.server is not None:
        os.environ['IMAP_SERVER'] = args.server
    if args.port is not None:
        os.environ['IMAP_PORT'] = str(args.port)
    if args.max_emails is not None:
        os.environ['MAX_EMAILS'] = str(args.max_emails)
    if args.output_dir is not None:
        os.environ['OUTPUT_DIR'] = args.output_dir
    if args.unread:
        os.environ['EMAIL_SEARCH_CRITERIA'] = 'UNSEEN'
    if args.gmail_categories is not None:
        os.environ['GMAIL_CATEGORIES'] = args.gmail_categories
    if args.fast_mode:
        os.environ['FAST_MODE'] = '1'
    
    main()