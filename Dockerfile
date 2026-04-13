FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY tools/ tools/
COPY .streamlit/ .streamlit/
COPY CLAUDE.md .

# Data files — these get copied from local .tmp/ at build time
# Only the two JSON files needed (no Chroma DB)
COPY .tmp/assets_with_text.json .tmp/assets_with_text.json
COPY .tmp/show_briefs.json .tmp/show_briefs.json

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "tools/chatbot_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
