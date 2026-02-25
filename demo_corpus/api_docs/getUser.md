# API Spec: GET /getUser

## Rate Limiting
- Default: 300 RPS per tenant
- Over limit: 429 RATE_LIMITED

## Timeout
- Client timeout: 1.5s
- Upstream timeout: 500ms
