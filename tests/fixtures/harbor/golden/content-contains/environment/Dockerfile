FROM python:3.12-slim

RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl bash \
      && rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
      apk add --no-cache git ca-certificates curl bash; \
    fi

RUN mkdir -p /app
WORKDIR /app

RUN git init -q -b main /app \
    && git -C /app config user.email "eval@agent-skill-eval.local" \
    && git -C /app config user.name "agent-skill-eval" \
    && git -C /app add -A \
    && GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000" \
       GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000" \
       git -C /app commit -q --allow-empty -m "baseline" \
    && git -C /app tag baseline \
    && git -C /app rev-parse HEAD > /app/.git/eval-baseline
