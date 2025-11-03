# Dockerfile for AI-Agent Agroforestry Dashboard
FROM python:3.10-slim

# System deps (for pygam, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gfortran libatlas-base-dev liblapack-dev     && rm -rf /var/lib/apt/lists/*

# Working dir
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY ai_agent_af_dashboard.py ./

# Streamlit config to run inside container
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose Streamlit default port
EXPOSE 8501

# Entrypoint
CMD streamlit run ai_agent_af_dashboard.py --server.port=$PORT --server.address=0.0.0.0
