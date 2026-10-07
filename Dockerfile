FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir -r requirements.txt
COPY sources.yaml podcasts.yaml ./
# Keys (OPENROUTER_API_KEY, COSMOS_KEY) come from the host's secrets; nothing secret is in
# the image. The web app serves from the source tree (static files sit beside app.py).
# `qdrss-ingest` runs the same image as a job, as does a one-off backfill (scripts/).
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 QDRSS_HOST=0.0.0.0 QDRSS_PORT=8080 QDRSS_SOURCES=sources.yaml
EXPOSE 8080
CMD ["python", "-c", "from qdrss.app import main; main()"]
