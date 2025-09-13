# Contributing to EmailAnalyser

Thanks for your interest in contributing! This document outlines how to get set up, submit changes, and keep users safe from accidentally sharing personal data.

## Getting Started

1. Fork the repo and clone locally.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy the config template:
   ```bash
   cp config.example.ini config.ini
   ```
4. Set environment variables for IMAP runs (recommended):
   ```bash
   export EMAIL_ADDRESS="you@example.com"
   export EMAIL_PASSWORD="your_app_password"
   ```

## Development Guidelines

- Keep `FETCH_WORKERS=1` by default (IMAP servers are often not thread-safe).
- Prefer env-based config; never hardcode credentials in code or tests.
- For Gmail, consider `GMAIL_CATEGORIES` and `GMAIL_PRIMARY_STRICT` to scope runs.
- Use `LIGHT_FETCH=0` for robust parsing when debugging header/body issues.
- Tests that touch IMAP must mock `imaplib.IMAP4_SSL` — do not run against real mailboxes.

## PII Safety (Very Important)

- Do not commit any outputs under `email_analysis_output/`.
- Do not commit `email_analyzer.log` or `config.ini`.
- Do not include real email addresses, subjects, or message text in issues/PRs.
- Before pushing, verify `.gitignore` excludes logs/outputs and configs.

## Testing

Run unit tests:
```bash
python -m unittest discover -s tests
```

If adding new parsing logic, please:
- Add or update tests under `tests/` using mocks.
- Include at least one fixture or inline RFC822 snippet (anonymized) for header parsing.

## Submitting Changes

1. Create a branch for your change.
2. Write clear commit messages and keep PRs focused.
3. Ensure tests pass locally.
4. Open a PR with:
   - A summary of the change and rationale.
   - A note about any user-facing behavior changes.
   - Confirmation you did not include PII.

## Code Style

- Keep changes minimal and focused.
- Match existing code style in `main.py`.
- Prefer readability and robustness over premature optimization.

Thanks for helping make EmailAnalyser better and safer!
