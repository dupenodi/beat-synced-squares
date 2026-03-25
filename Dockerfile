FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY api ./api
COPY web ./web

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
