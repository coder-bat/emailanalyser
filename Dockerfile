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
RUN npm run build

# Python backend stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask for API server
RUN pip install flask flask-cors gunicorn

# Copy Python source code
COPY *.py ./
COPY tests/ ./tests/

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