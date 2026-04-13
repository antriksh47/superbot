FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY server.py .
COPY tools/data_tools.py tools/data_tools.py
COPY tools/chatbot_app.py tools/chatbot_app.py
COPY static/ static/
COPY .streamlit/ .streamlit/

# Data files
COPY .tmp/assets_with_text.json .tmp/assets_with_text.json
COPY .tmp/show_briefs.json .tmp/show_briefs.json

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
