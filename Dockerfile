# Use Python 3.11 (slim version to save space and RAM)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# (build-essential is needed if any library needs to be compiled)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the app code
COPY . .

# Expose the standard Streamlit port
EXPOSE 8501

# Healthcheck: Tells Coolify if the app is alive or dead
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]