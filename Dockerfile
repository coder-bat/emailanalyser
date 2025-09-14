# Multi-stage build for EmailAnalyser with React frontend
FROM node:18-alpine AS frontend-build

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ ./

# Build frontend for production
ENV CI=false
RUN npm run build

# Python backend stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .

# Install with trusted hosts to handle SSL issues
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Install Flask for API server
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org flask flask-cors gunicorn

# Copy Python source code
COPY *.py ./
COPY tests/ ./tests/ 2>/dev/null || true

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build/

# Create output directory
RUN mkdir -p email_analysis_output

# Set environment variables
ENV PYTHONPATH=/app
ENV OUTPUT_DIR=/app/email_analysis_output
ENV API_PORT=5000
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Create non-root user for security
RUN useradd -m -u 1000 emailuser && chown -R emailuser:emailuser /app
USER emailuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the API server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "api_server:app"]