#!/usr/bin/env python3
"""
Lightweight demo runner for EmailAnalyser
Creates sample emails, runs simple categorization and importance scoring,
and prints a short summary. Does not require external libraries.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Dict
import json


@dataclass
class EmailMessage:
    uid: str
    subject: str
    sender: str
    recipient: str
    date: datetime
    body: str
    attachments: List[str] = field(default_factory=list)
    category: str = 'other'
    importance_score: float = 0.0


class SimpleConfig:
    def __init__(self):
        self.categories = {
            'promotional': ['sale', 'discount', 'offer', 'deal', 'free'],
            'work': ['meeting', 'project', 'deadline', 'report', 'task'],
            'personal': ['family', 'friend', 'birthday', 'congratulations', 'thank'],
            'newsletter': ['newsletter', 'digest', 'update', 'subscribe'],
            'spam': ['winner', 'claim', 'urgent', 'prize']
        }
        self.urgent_keywords = ['urgent', 'asap', 'immediate', 'critical']
        self.important_keywords = ['important', 'priority', 'attention']


class SimpleCategorizer:
    def __init__(self, config: SimpleConfig):
        self.config = config

    def categorize(self, email: EmailMessage) -> str:
        text = (email.subject + ' ' + email.body).lower()
        scores = {k: 0 for k in self.config.categories}
        for cat, kws in self.config.categories.items():
            for kw in kws:
                if kw in text:
                    scores[cat] += 1

        # sender hints
        if 'noreply' in email.sender.lower() or 'newsletter' in email.sender.lower():
            scores['newsletter'] += 2

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = 'other'
        email.category = best
        return best


class SimpleImportanceScorer:
    def __init__(self, config: SimpleConfig):
        self.config = config

    def score(self, email: EmailMessage) -> float:
        score = 0.0
        text = (email.subject + ' ' + email.body).lower()
        for kw in self.config.urgent_keywords:
            if kw in text:
                score += 0.5
                break
        for kw in self.config.important_keywords:
            if kw in text:
                score += 0.3
                break
        # attachments add a bit
        if email.attachments:
            score += 0.1

        # recency boost
        days = (datetime.now() - email.date).days
        if days <= 1:
            score += 0.2
        elif days <= 7:
            score += 0.1

        email.importance_score = min(score, 1.0)
        return email.importance_score


def build_sample_emails() -> List[EmailMessage]:
    now = datetime.now()
    samples = [
        EmailMessage('1', 'Big Sale this weekend', 'promo@shop.example.com', 'you@example.com', now - timedelta(days=2), 'Huge discount, limited time offer!'),
        EmailMessage('2', 'Project update', 'alice@company.com', 'you@example.com', now - timedelta(hours=5), 'Please see attached report for the project deadline.'),
        EmailMessage('3', 'Family reunion', 'mom@example.com', 'you@example.com', now - timedelta(days=30), 'Looking forward to seeing you for the birthday.'),
        EmailMessage('4', 'Weekly Newsletter', 'news@newsletter.com', 'you@example.com', now - timedelta(days=1), 'This week in news...'),
        EmailMessage('5', 'You are a winner!', 'scam@spam.com', 'you@example.com', now - timedelta(days=100), 'Claim your prize now!'),
        EmailMessage('6', 'Immediate: Action required', 'boss@company.com', 'you@example.com', now - timedelta(hours=2), 'This is urgent, please respond ASAP.'),
    ]
    # add an attachment to one email
    samples[1].attachments.append('report.pdf')
    return samples


def main():
    config = SimpleConfig()
    categorizer = SimpleCategorizer(config)
    scorer = SimpleImportanceScorer(config)

    emails = build_sample_emails()

    for e in emails:
        categorizer.categorize(e)
        scorer.score(e)

    # Summary
    print('\nDemo Email Analysis Summary')
    print('---------------------------------')
    print(f'Total emails: {len(emails)}')
    cats = Counter(e.category for e in emails)
    print('Categories:')
    for k, v in cats.items():
        print(f'  - {k}: {v}')

    important = [e for e in emails if e.importance_score >= 0.5]
    print(f'Important emails (score>=0.5): {len(important)}')
    for e in sorted(important, key=lambda x: x.importance_score, reverse=True):
        print(f"  - [{e.importance_score:.2f}] {e.subject} (from: {e.sender})")

    # Save a small JSON output
    out = [
        {
            'uid': e.uid,
            'subject': e.subject,
            'sender': e.sender,
            'date': e.date.isoformat(),
            'category': e.category,
            'importance_score': e.importance_score
        }
        for e in emails
    ]
    with open('demo_output.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('\nSaved demo_output.json')


if __name__ == '__main__':
    main()
