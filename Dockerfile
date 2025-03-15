FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /venv /venv
COPY --from=builder /app /app

ENV PATH="/venv/bin:$PATH"

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]