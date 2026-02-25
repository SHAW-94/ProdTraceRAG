# API Spec: POST /createOrder

## Auth
- Requires service-to-service token (S2S)

## Rate Limiting
- Default: 100 RPS per tenant
- Burst: 200 RPS for 10 seconds
- Over limit returns HTTP 429 with error code RATE_LIMITED

## Timeout
- Client timeout: 2s
- Upstream timeout: 800ms

## Errors
- 400 INVALID_ARGUMENT
- 401 UNAUTHORIZED
- 429 RATE_LIMITED
- 500 INTERNAL
