# EmailAnalyser

A practical, local-first IMAP email analysis tool that focuses on actionable insights — not just charts. It fetches messages, categorizes them (with Gmail category support), computes sender-level stats and importance scores, and produces CSVs and a concise text report. It highlights senders to consider unsubscribing/deleting and likely important senders.

Key components live in `main.py`. A small offline demo runner is available as `run_demo.py`.

---

### Features

- Sender-level analytics with normalized addresses (groups by canonical `sender_email`).
- Actionables out-of-the-box:
  - Senders to consider deleting/unsubscribing (frequency + newsletter ratio + low average importance).
  - Important senders (high average importance or heuristics like finance/notifications).
- Gmail category filtering using `X-GM-RAW` including Primary-only mode (strict or heuristic).
- Newest-first limiting and batching for performance on large mailboxes.
- Two fetch modes for reliability/perf:
  - Light fetch: headers + limited text + BODYSTRUCTURE.
  - Full fetch: full RFC822 fallback for robust parsing of headers/subjects.
- Data-first outputs: CSVs and a human-friendly text report, visuals off by default.

---

### Requirements

- Python 3.10+ (tested on 3.10/3.13)
- An IMAP account. For Gmail: IMAP enabled and an App Password recommended.

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the offline demo (no credentials):

```bash
python3 run_demo.py
```

---

### How to Run (env-based)

This tool is configured primarily via environment variables. Typical runs:

```bash
# Strict Primary-only analysis, newest 500, stable serial full-fetch
FETCH_WORKERS=1 \
SKIP_HEADER_BATCH=1 \
FAST_MODE=0 \
LIGHT_FETCH=0 \
GMAIL_CATEGORIES="Primary" \
GMAIL_PRIMARY_STRICT=1 \
MAX_EMAILS=500 \
EMAIL_ADDRESS="you@gmail.com" \
EMAIL_PASSWORD="your_app_password" \
python3 main.py
```

Header-only quick inventory (fast):

```bash
FAST_MODE=1 \
GMAIL_CATEGORIES="Primary" \
GMAIL_PRIMARY_STRICT=1 \
MAX_EMAILS=300 \
EMAIL_ADDRESS="you@gmail.com" \
EMAIL_PASSWORD="your_app_password" \
python3 main.py
```

Unread-only triage:

```bash
EMAIL_SEARCH_CRITERIA=UNSEEN \
GMAIL_CATEGORIES="Promotions" \
MAX_EMAILS=200 \
EMAIL_ADDRESS="you@gmail.com" \
EMAIL_PASSWORD="your_app_password" \
python3 main.py
```

---

### Configuration Reference

Credentials and server
- `EMAIL_ADDRESS`: IMAP username/email.
- `EMAIL_PASSWORD`: IMAP/app password.
- `IMAP_SERVER`/`IMAP_PORT`: Defaults `imap.gmail.com` / `993`.
- `OUTPUT_DIR`: Directory for outputs (default `email_analysis_output`).

Scope and filtering
- `MAX_EMAILS`: Limit, applied newest-first.
- `EMAIL_SEARCH_CRITERIA`: IMAP SEARCH expression (e.g., `ALL`, `UNSEEN`, `SINCE 01-Jan-2025`).
- `GMAIL_CATEGORIES`: Comma list of `Primary,Promotions,Social,Updates,Forums`.
- `GMAIL_PRIMARY_STRICT`: `1` to use Gmail’s native `in:inbox category:primary`; `0` to approximate Primary as Inbox minus Promotions/Social/Updates/Forums.
- `GMAIL_COMBINED`: `1` (default) to OR categories in a single query; `0` falls back to per-category queries.

Performance and reliability
- `FAST_MODE`: `1` to write header-only CSV and exit early.
- `LIGHT_FETCH`: `1` for light mode (headers + limited body + structure); set `0` for full RFC822 fetch (most robust).
- `FORCE_FULL_FETCH`: `1` also forces full RFC822 fetch.
- `SKIP_HEADER_BATCH`: `1` to skip the batched header phase (go straight to body fetch). Useful if header batch is slow or inconsistent.
- `FETCH_WORKERS`: Number of parallel body fetch workers (default `1`). IMAP connections are not thread-safe; keep at `1` unless you know your server tolerates concurrency.
- `VIS_ENABLED`: `1` to enable visualization exports; disabled by default.

Importance and actionables
- `IMPORTANT_THRESHOLD`: Float (0–1) importance cut-off used in reports (default `0.6`).
- Additional heuristics are configurable via `config.ini` under `[IMPORTANCE]` and `[ACTIONS]` (see below).

Config file (`config.ini`)
- Defaults live in `config.ini` and are loaded by `main.py`. Environment variables override config values.
- Relevant sections:
  - `[EMAIL]`: server, port, folder, batch sizes, `light_fetch` default, etc.
  - `[IMPORTANCE]`: VIP domains, keywords, `important_threshold`.
  - `[ACTIONS]` (optional): thresholds for deletion candidates and important senders.

---

### Gmail Category Filtering

- Uses Gmail’s `X-GM-RAW` under the hood.
- `GMAIL_CATEGORIES="Primary"` with `GMAIL_PRIMARY_STRICT=1` uses `in:inbox category:primary`.
- Non-strict Primary uses `in:inbox -category:social -category:promotions -category:updates -category:forums`.
- For multiple categories, the tool builds a combined OR query when `GMAIL_COMBINED=1`; otherwise it unions per-category searches.

---

### Outputs

Written to `email_analysis_output/` (or `OUTPUT_DIR`).
- `email_report_YYYYMMDD_HHMMSS.txt`: Human-friendly summary (counts, category breakdown, top senders, recommendations).
- `email_data_YYYYMMDD_HHMMSS.csv`: Row per email with `date, sender, sender_email, subject, category, importance_score, has_attachments`.
- `sender_stats.csv`: Aggregated per normalized sender (`sender_key`, label, totals, important counts).
- `senders_to_delete.csv`: Candidates to unsubscribe/delete (count, avg_importance, newsletter_pct).
- `important_senders.csv`: Senders with consistently high importance.
- `summary.json`: Quick totals (`total_senders`, counts of actionables).
- Optional visuals (if `VIS_ENABLED=1`): PNG charts and a Plotly HTML dashboard.

---

### How it works (brief)

- Fetch phase: newest-first UIDs; optional batched header fetch; light vs full RFC822 fetch.
- Parsing: robust header decoding; normalized `sender_email`; safe date normalization to naive UTC.
- Categorization: respects Gmail labels when available, falls back to keyword heuristics; optional NLP sentiment.
- Scoring: sender frequency, urgency keywords, recency, relevance windows, and attachments into a 0–1 importance score.
- Actionables: per-sender aggregation to recommend deletes/unsubscribes and flag important senders.

---

### Troubleshooting

- Auth errors (Gmail): ensure IMAP is enabled and use an App Password.
- Blank headers/subjects: set `LIGHT_FETCH=0` (full RFC822) and consider `SKIP_HEADER_BATCH=1`.
- Slow runs on large inboxes: set `GMAIL_CATEGORIES`, `EMAIL_SEARCH_CRITERIA=UNSEEN`, and a smaller `MAX_EMAILS`.
- Empty results with Primary: confirm messages actually live in Primary; try non-strict (`GMAIL_PRIMARY_STRICT=0`).
- Concurrency issues/segfaults: keep `FETCH_WORKERS=1`.

---

### Security

- Prefer environment variables for credentials. Never commit secrets.
- For shared machines, unset variables when done: `unset EMAIL_PASSWORD`.

---

### Demo vs. Live

- `run_demo.py` produces synthetic outputs without network access — useful to validate the pipeline and visuals.
- `main.py` requires IMAP access and installed dependencies.

---

### Examples

Strict Primary-only, 500 newest, serial and robust:
```bash
FETCH_WORKERS=1 SKIP_HEADER_BATCH=1 FAST_MODE=0 LIGHT_FETCH=0 \
GMAIL_CATEGORIES="Primary" GMAIL_PRIMARY_STRICT=1 MAX_EMAILS=500 \
EMAIL_ADDRESS="you@gmail.com" EMAIL_PASSWORD="your_app_password" \
python3 main.py
```

Header-only quick scan (no bodies):
```bash
FAST_MODE=1 GMAIL_CATEGORIES="Primary" GMAIL_PRIMARY_STRICT=1 MAX_EMAILS=300 \
EMAIL_ADDRESS="you@gmail.com" EMAIL_PASSWORD="your_app_password" \
python3 main.py
```

---

## Quick start

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (for full functionality):

```bash
pip install -r requirements.txt
```

3. Run the demo (no external network or credentials required):

```bash
python3 run_demo.py
```

4. Run the full analyzer against an IMAP server (Gmail example):

```bash
python3 main.py --email you@gmail.com --server imap.gmail.com --max-emails 500 --output-dir email_analysis_output
```

Note: For Gmail you will likely need an app-specific password and IMAP enabled in your account settings.

## CLI options (summary)

- `--email`: Email address to analyse.
- `--server`: IMAP server (default `imap.gmail.com`).
- `--port`: IMAP port (default `993`).
- `--max-emails`: Maximum number of emails to fetch and process.
- `--unread`: Only fetch unread (`UNSEEN`) messages.
- `--gmail-categories`: Comma-separated Gmail categories to restrict the search to (e.g. `Primary,Promotions,Social,Updates`).
- `--output-dir`: Directory to write reports and visualizations.

Examples:

```bash
# Fetch unread promotions only (limit 200)
python3 main.py --email you@gmail.com --max-emails 200 --unread --gmail-categories Promotions

# Fetch primary + social, but include read and unread
python3 main.py --email you@gmail.com --max-emails 300 --gmail-categories Primary,Social
```

## How Gmail category filtering works

Gmail assigns messages to categories such as `Primary`, `Promotions`, `Social`, and `Updates`. The script supports filtering by these categories using Gmail's extended IMAP search token `X-GM-RAW`.

Implementation notes and behavior:

- When you pass `--gmail-categories`, the connector will run one IMAP search per category and combine results. Example internal query: `X-GM-RAW "category:promotions UNSEEN"` (if `--unread` also set).
- The script preserves the server-returned order per-category and deduplicates message IDs across categories.
- If `X-GM-RAW` is not supported (non-Gmail server), the connector attempts a fallback IMAP SEARCH using the token and criteria (this is imperfect because standard IMAP servers don't understand Gmail category tokens).
- `--max-emails` is applied after the category-based filtering: the pipeline will only fetch up to `--max-emails` message IDs from the combined, ordered results. This ensures your limit refers to the specific category set you requested.

Tradeoffs and tips:

- Per-category searches are reliable for Gmail and simple to reason about. Constructing a single complex OR query might reduce roundtrips but can be harder to get right across IMAP servers.
- If you want the absolute newest messages across multiple categories, consider running a single `X-GM-RAW` query with combined tokens (not currently implemented) or fetch from a single category and then expand if needed.
- Using `--unread` combined with categories is efficient: it reduces the number of fetched messages because the server will only return matching unseen messages in those categories.

## How changing the IMAP search query helps

The IMAP `SEARCH` (and Gmail's `X-GM-RAW`) controls which message UIDs the server returns. Adjusting the search query helps in these ways:

- Performance: Narrower queries return fewer UIDs, reducing network transfer and processing time.
- Precision: You can target only important subsets (e.g., unread Promotions, or only Primary messages) to focus analysis and keep results relevant.
- Cost of scanning: Large mailboxes can contain hundreds of thousands of messages — narrowing by category and `UNSEEN` prevents the script from trying to iterate through the entire mailbox.

Common useful queries:

- `ALL` — everything in the selected mailbox (default). Use only for small mailboxes or during debugging.
- `UNSEEN` — unread messages only (good when you want to triage new email).
- `SINCE 01-Jan-2025` — messages since a date (IMAP date format). Combine with categories if you want recent category-scoped messages.
- `X-GM-RAW "category:promotions UNSEEN"` — Gmail-only: unseen promotions.

If you need more advanced filtering (attachments, specific senders), you can set `EMAIL_SEARCH_CRITERIA` environment variable to any IMAP-search expression or extend `main.py` to accept a `--query` parameter.

## Security and credentials

- The script reads credentials from environment variables `EMAIL_ADDRESS` and `EMAIL_PASSWORD` when available. If `EMAIL_PASSWORD` is not set, it will prompt for a password at runtime.
- For Gmail, create an app-specific password (preferably) or enable the appropriate OAuth flow (not implemented here). Do not commit credentials to source control.

## Troubleshooting

- Authentication failures: re-check the email, password, and whether IMAP is enabled. For Gmail, confirm app-specific password or OAuth settings.
- No messages returned for expected categories: verify the mailbox actually contains messages in those categories via the Gmail web UI; `X-GM-RAW` is Gmail-specific.
- `X-GM-RAW` failures (search error): fallback is attempted, but for full Gmail capability you must connect to `imap.gmail.com` and be authenticated successfully.

## Next steps and optional improvements

- Add a `--query` CLI argument to pass arbitrary IMAP search expressions directly.
- Implement a single combined `X-GM-RAW` OR query builder for advanced users (faster, fewer roundtrips).
- Add OAuth2 support for Gmail to avoid app-specific passwords.
- Add unit tests that mock IMAP server responses to validate category behavior without a real account.

---

### Configurable combined-search behavior

The code supports building a single combined `X-GM-RAW` query (ORed categories) to reduce IMAP roundtrips. By default this is enabled. To disable and force per-category searches, set the environment variable:

```bash
export GMAIL_COMBINED=0
```

When combined is enabled the script attempts a single `X-GM-RAW` query like `X-GM-RAW "(category:promotions OR category:social) UNSEEN"`. If that combined query fails the connector falls back to per-category searches.

# EmailAnalyser

Lightweight email analysis tool. `main.py` contains a full feature set but requires several Python packages and an IMAP account. For quick testing there's a `run_demo.py` which runs without external deps.

Setup (recommended inside a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Running the demo (no external services required):

```bash
python3 run_demo.py
```

Running the full analyzer (requires IMAP access and installed deps):

```bash
python3 main.py --email "you@example.com" --server "imap.gmail.com" --max-emails 1000 --output-dir analysis_output
```

Notes:
- For Gmail, create an app password and enable IMAP in your account settings.
- `main.py` may require small refactors before running against a real inbox — open an issue or ask for help
