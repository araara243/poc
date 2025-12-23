# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (to leverage Docker layer caching)
COPY requirements.txt .
COPY packages.txt .

# Install system dependencies FROM packages.txt
RUN apt-get update && xargs -a packages.txt apt-get install -y \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "dashboard/app_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
