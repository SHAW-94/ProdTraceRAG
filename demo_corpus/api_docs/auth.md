# Auth Spec: Service-to-Service (S2S)

## Tokens
- JWT-based, rotated every 24h

## Common failures
- 401 UNAUTHORIZED when token expired
- 403 FORBIDDEN when scope missing
