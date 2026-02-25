# API Spec: POST /uploadReceipt

## Limits
- Max payload: 5MB
- Content-Type must be image/*

## Errors
- 413 PAYLOAD_TOO_LARGE
- 415 UNSUPPORTED_MEDIA_TYPE
