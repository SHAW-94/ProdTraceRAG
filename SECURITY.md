# Security Policy

## Scope

This project can run in two modes:

- **Local / trusted-network demo mode** (default for backwards compatibility)
- **Internet-facing hardened mode** (recommended for production or public exposure)

This patch adds authentication, IP allowlists, ingestion path allowlists, and redaction-by-default for config/traces.

## Minimum Secure Deployment Settings

Set at least the following environment variables before exposing the service publicly:

```bash
export AUTH_REQUIRED=1
export PRODTRACERAG_API_TOKEN="replace-with-long-random-token"
export PRODTRACERAG_ADMIN_TOKEN="replace-with-different-long-random-token"

# Optional but strongly recommended
export PUBLIC_ALLOWLIST_CIDRS="0.0.0.0/0"
export ADMIN_ALLOWLIST_CIDRS="203.0.113.10/32,198.51.100.0/24"
export TRUST_PROXY_HEADERS=1   # only when running behind a trusted reverse proxy

# Local ingestion safety
export INGEST_REQUIRE_ALLOWLIST=1
export INGEST_ALLOWED_ROOTS="/srv/prodtracerag/corpus"

# Reduce accidental information exposure
export EXPOSE_CITATION_SNIPPETS=0
export TRACE_REDACT_ON_WRITE=1
export TRACE_ALLOW_RAW_EXPORT=0
export ENABLE_DOCS=0
```

## Endpoint Access Model

- `POST /ask`: public token (or admin token)
- `GET /config`: public by default (redacted); `?scope=admin` requires admin token
- `POST /ingest_local`, `POST /reset_index`, `POST /set_config`, `GET /traces/*`, `GET /chunk`: admin token

## Supported Hardening Controls

- **Bearer token auth** (`AUTH_REQUIRED`, `PRODTRACERAG_API_TOKEN`, `PRODTRACERAG_ADMIN_TOKEN`)
- **IP allowlists** (`PUBLIC_ALLOWLIST_CIDRS`, `ADMIN_ALLOWLIST_CIDRS`)
- **Trusted proxy mode** (`TRUST_PROXY_HEADERS`)
- **Local ingestion path allowlist** (`INGEST_REQUIRE_ALLOWLIST`, `INGEST_ALLOWED_ROOTS`)
- **Redaction by default** for trace/config outputs
- **Docs can be disabled** (`ENABLE_DOCS=0`)

## Vulnerability Reporting

Please replace this section with your real contact address before publishing.

- Security contact: `security@your-domain.example`
- Please include:
  - affected version / commit
  - reproduction steps
  - impact assessment
  - whether the issue is publicly disclosed
