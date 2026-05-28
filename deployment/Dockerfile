# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose port for the API
EXPOSE 8000

# Expose port for Streamlit dashboard
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Default command (can be overridden)
CMD ["python", "-m", "src.dashboard.app"]