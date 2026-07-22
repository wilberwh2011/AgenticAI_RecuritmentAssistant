# Local Langfuse (self-host, demo use)

## Start it
```bash
docker compose up -d
or
docker compose -f observability/docker-compose.langfuse.yml up -d
```
First boot takes 2-3 minutes (Postgres/ClickHouse migrations). Watch readiness:
```bash
docker compose logs -f langfuse-web
```
Ready when you see the web container log "Ready".

## Log in
UI: http://localhost:3000
- Email: `demo@example.com`
- Password: `demopassword123`

(Both are set via `LANGFUSE_INIT_USER_*` in `.env` — change before sharing this beyond your machine.)

## Point your app at it
The `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `_SECRET_KEY` in `.env` are pre-created for you —
no need to click through the UI to generate API keys. In your Python app:

```bash
pip install langfuse
```
```python
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-local-demo"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-local-demo"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
```

If you're using the OTel path instead of the Langfuse SDK directly, Langfuse also exposes
an OTLP endpoint at `http://localhost:3000/api/public/otel` — same traces, OpenTelemetry protocol.

## Stop it
```bash
docker compose down       # keep data
or 
docker compose  -f observability/docker-compose.langfuse.yml down
docker compose down -v    # wipe volumes (fresh start)
```

## Notes
- This is the official 6-container v3 stack (Postgres, ClickHouse, Redis, MinIO, web, worker) —
  heavier than v2's single-Postgres setup, but it's what's currently supported for self-host.
- `.env` holds real generated secrets (NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY, REDIS_AUTH).
  Fine for local/demo use; don't reuse these values anywhere internet-facing.
- Do not commit `.env` to git.
