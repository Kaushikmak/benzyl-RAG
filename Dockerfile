FROM python:3.11-slim

WORKDIR /app

# Install system dependencies, including curl for Ollama installation and Node.js for Next.js
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x scripts/docker-entrypoint.sh

# Expose ports for FastAPI (8000) and Next.js (3000)
EXPOSE 8000
EXPOSE 3000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["python", "main.py", "web"]