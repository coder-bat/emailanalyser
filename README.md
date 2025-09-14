# EmailAnalyser

A comprehensive, local-first IMAP email analysis tool with a modern React frontend. It fetches messages, categorizes them (with Gmail category support), computes sender-level stats and importance scores, and produces interactive dashboards, CSVs and detailed reports.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Run with Docker
docker run -p 5000:5000 ghcr.io/coder-bat/emailanalyser:latest

# Visit http://localhost:5000
```

### Option 2: Docker Compose

```bash
# Download docker-compose.yml
wget https://raw.githubusercontent.com/coder-bat/emailanalyser/main/docker-compose.yml

# Start the service
docker-compose up -d

# Visit http://localhost:5000
```

### Option 3: Local Development

```bash
# Clone the repository
git clone https://github.com/coder-bat/emailanalyser.git
cd emailanalyser

# Run setup script
./setup.sh

# Start the API server
python api_server.py

# Visit http://localhost:5000
```

## 🌐 Live Demo

Visit the **[GitHub Pages Demo](https://coder-bat.github.io/emailanalyser/)** to see the frontend interface and Docker deployment instructions.

## ✨ Features

### Frontend Dashboard
- 📊 **Interactive Analytics**: Charts and visualizations for email patterns
- 📈 **Real-time Insights**: Email volume, categories, and sender statistics  
- 🎯 **Actionable Recommendations**: Senders to unsubscribe/delete and important contacts
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Live Data Updates**: Connects to Python backend via REST API

### Backend Analysis Engine
- 🔍 **Sender-level Analytics**: Normalized addresses with canonical grouping
- 🎯 **Smart Recommendations**: 
  - Senders to consider deleting/unsubscribing (frequency + newsletter ratio + low importance)
  - Important senders (high importance or heuristics like finance/notifications)
- 📧 **Gmail Integration**: Category filtering using `X-GM-RAW` including Primary-only mode
- ⚡ **Performance Optimized**: Newest-first limiting and batching for large mailboxes
- 🔄 **Flexible Fetch Modes**: Light fetch (headers + limited text) or full RFC822 for robust parsing
- 📊 **Rich Outputs**: Interactive dashboard, CSV exports, text reports, and optional visualizations
 - 🚀 **On-Demand Runs**: Trigger fresh analyses from the UI (no manual shell needed)
 - 🔁 **Single Active Job Guard**: Prevents duplicate heavy analyses; second request reuses current job
 - 📡 **Streaming Progress Heuristics**: Backend parses `main.py` stage logs to emit % completion
 - 🔐 **Ephemeral Credential Injection**: Password passed only as env var to subprocess; never persisted in job data
 - 🛡️ **Newest-First Limiting**: Large mailboxes trimmed to newest `MAX_EMAILS` before heavy processing
 - 🧠 **Adaptive Fetch Modes**: Light fetch first, upgrade selectively if headers incomplete

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React         │    │   Flask API     │    │   Python        │
│   Frontend      │────│   Server        │────│   Analysis      │
│   (Port 3000)   │    │   (Port 5000)   │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
    ┌───▼────┐              ┌────▼────┐              ┌────▼────┐
    │ Static │              │   API   │              │  Email  │
    │ Assets │              │ Routes  │              │  Data   │
    └────────┘              └─────────┘              └─────────┘
```

### Components

1. **React Frontend** (`/frontend`): Modern TypeScript React app with Material-UI
2. **Flask API Server** (`api_server.py`): RESTful API bridge between frontend and analysis engine
3. **Python Analysis Engine** (`main.py`): Core email processing and analysis logic
4. **Job Runner (in `api_server.py`)**: Launches `main.py` subprocess per analysis with progress tracking

### Job Execution Flow
1. User opens "Run Analysis" dialog in the frontend and supplies email, options, (app) password.
2. Frontend calls `POST /api/run-analysis`.
3. API creates a job id (or reuses the running one) and spawns `python main.py` in a thread.
4. Server tails stdout; regex-based milestones update in-memory job `progress`.
5. Frontend polls `GET /api/analysis-status/<job_id>` every few seconds to update a linear progress bar.
6. Upon completion, frontend auto-refreshes datasets (emails, sender stats, recommendations).

The job state currently lives in-process (memory). Using multiple Gunicorn workers will create isolated job maps; for consistent progress reporting run with a single worker (see Recommendation below).

## 🛠️ Configuration

### Environment Variables

#### Email Connection
- `EMAIL_ADDRESS`: IMAP username/email
- `EMAIL_PASSWORD`: IMAP/app password  
- `IMAP_SERVER`/`IMAP_PORT`: Server details (defaults: `imap.gmail.com:993`)

#### Analysis Scope
- `MAX_EMAILS`: Limit, applied newest-first (default: 1000)
- `EMAIL_SEARCH_CRITERIA`: IMAP SEARCH expression (e.g., `ALL`, `UNSEEN`, `SINCE 01-Jan-2025`)
- `GMAIL_CATEGORIES`: Comma list of `Primary,Promotions,Social,Updates,Forums`
- `GMAIL_PRIMARY_STRICT`: `1` for native `in:inbox category:primary`; `0` for approximation

#### Performance & Reliability  
- `FAST_MODE`: `1` for header-only CSV and early exit
- `LIGHT_FETCH`: `1` for light mode; `0` for full RFC822 fetch
- `SKIP_HEADER_BATCH`: `1` to skip batched header phase
- `FETCH_WORKERS`: Parallel workers (default: `1` for stability)

#### API Server
- `API_PORT`: Flask server port (default: 5000)
- `OUTPUT_DIR`: Analysis output directory
- `HOST_API_PORT`: Host port mapping used in docker-compose (e.g. `5050:5000`)

#### Job / Progress Internals (no need to change normally)
- Progress relies on `main.py` emitting stage markers like `[1/7]`, `[2/7]`, etc.
- Additional phrases such as `Performing batched header fetch`, `Header fetch completed`, `Will fetch full bodies for`, `Retrieved X emails`, and `Analysis Complete` refine percentages.

> NOTE: Changing those log lines in `main.py` will affect progress resolution; keep markers if you customize.

### Example Usage

```bash
# Gmail Primary-only analysis, newest 500 emails
FETCH_WORKERS=1 \
GMAIL_CATEGORIES="Primary" \
GMAIL_PRIMARY_STRICT=1 \
MAX_EMAILS=500 \
EMAIL_ADDRESS="you@gmail.com" \
EMAIL_PASSWORD="your_app_password" \
python main.py
```

## 📊 API Endpoints

The Flask API server provides these endpoints:

- `GET /api/summary` - Analysis overview statistics
- `GET /api/emails` - Email data with pagination
- `GET /api/sender-stats` - Per-sender statistics  
- `GET /api/senders-to-delete` - Cleanup recommendations
- `GET /api/important-senders` - High-priority contacts
- `GET /api/patterns` - Time and category patterns
- `POST /api/run-analysis` - Trigger new analysis
- `GET /api/analysis-status/<job_id>` - Poll current job (status, progress)
- `GET /health` - Health check endpoint

## 🐳 Docker Deployment  

### GitHub Container Registry

Images are automatically built and published to GitHub Container Registry:

```bash
# Pull latest image
docker pull ghcr.io/coder-bat/emailanalyser:latest

# Run with custom configuration
docker run -p 5000:5000 \
  -e EMAIL_ADDRESS=your-email@gmail.com \
  -e EMAIL_PASSWORD=your-app-password \
  -e MAX_EMAILS=1000 \
  ghcr.io/coder-bat/emailanalyser:latest
```

### Local Build

```bash
# Build image
docker build -t emailanalyser .

# Run container
docker run -p 5000:5000 emailanalyser
```

### Recommended Runtime (Single Worker)

Because job tracking is stored in memory, use a single Gunicorn worker for accurate progress:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 api_server:app
```

If you need concurrency, consider externalizing job state (Redis / DB) first.

## 🔄 CI/CD Pipeline

GitHub Actions automatically:

1. **Tests**: Run Python tests and frontend build
2. **Docker Build**: Create and push container images to GHCR  
3. **GitHub Pages**: Deploy frontend demo and documentation
4. **Security**: Scan dependencies and container images

## 📱 Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server  
npm start

# Build for production
npm run build

# Run tests
npm test
```

## 🔧 Backend Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run demo (no IMAP required)
python run_demo.py

# Run full analysis
python main.py --email you@gmail.com --max-emails 500

# Start API server
python api_server.py
```

## 📋 Requirements

- **Python**: 3.10+ (tested on 3.10/3.13)
- **Node.js**: 18+ for frontend development
- **Docker**: For containerized deployment
- **IMAP Account**: Gmail recommended with app password

## 🔒 Security

- Prefer environment variables for credentials
- Never commit secrets to source control
- Use Gmail app passwords instead of main password
- Run containers as non-root user
- Regular dependency updates via Dependabot
- Passwords are only set via environment for the subprocess and are not returned by any API endpoint.
- Avoid enabling multiple workers until job state is externalized.

### Optional Hardening Ideas
- Add TLS termination with a reverse proxy (Traefik / Caddy / Nginx)
- Mount a read-only config volume; separate writable `OUTPUT_DIR`
- Store secrets in Docker/Compose secrets instead of plain env vars
- Use `EMAIL_DATE_SINCE` (future enhancement) to minimize downloaded messages

## 🧪 Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|-----------|
| Progress bar stalls early | Stage log not yet emitted | Wait for next milestone; check container logs |
| `job not found` 404 | Multiple Gunicorn workers used | Run with `--workers 1` |
| 403 / CORS errors in browser | Hardcoded absolute API URL | Use relative paths (already default) |
| Very large "Search returned" number | IMAP ALL returns every id | Limit via `MAX_EMAILS` & date criteria |
| Repeated analysis submissions | User clicked multiple times | Guard now reuses active job (202 response) |
| Password persists in job status | (Should not happen) | Ensure password key excluded in job dict |

Check logs:
```bash
docker compose logs --tail=200 emailanalyser
```

Health endpoint:
```bash
curl -s http://localhost:5000/health | jq
```

## 🗺️ Roadmap (Planned Enhancements)

- Cancel / abort job endpoint
- Persistent job history (timestamped runs, durations, parameters)
- Date-based mailbox limiting (`EMAIL_DATE_SINCE`, `EMAIL_DATE_UNTIL`)
- External state store for multi-worker scaling
- WebSocket / SSE live progress streaming (reduce polling)
- Fine-grained per-phase timing metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Quick Links

- 🌐 **[Live Demo](https://coder-bat.github.io/emailanalyser/)**
- 🐳 **[Docker Images](https://github.com/coder-bat/emailanalyser/pkgs/container/emailanalyser)**  
- 📚 **[Documentation](https://github.com/coder-bat/emailanalyser/wiki)**
- 🐛 **[Issues](https://github.com/coder-bat/emailanalyser/issues)**
- 💬 **[Discussions](https://github.com/coder-bat/emailanalyser/discussions)**