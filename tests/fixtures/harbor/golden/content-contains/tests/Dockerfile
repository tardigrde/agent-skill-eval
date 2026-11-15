FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pydantic==2.13.4

WORKDIR /app
COPY . /tests/
