FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Systempakete installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential locales && \
    rm -rf /var/lib/apt/lists/* && \
    sed -i 's/\# de_CH.UTF-8 UTF-8/de_CH.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

WORKDIR /app

# Abhängigkeiten zuerst (Layer-Caching)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Projektcode + statische Dateien
COPY src ./src
COPY public ./public
COPY data ./data

# Port & Start
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.alesa_bot.web_api:app", "--host", "0.0.0.0", "--port", "8080"]
