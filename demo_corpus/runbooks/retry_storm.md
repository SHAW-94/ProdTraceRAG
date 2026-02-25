# Runbook: Retry Storm

## Symptoms
- QPS rises unexpectedly
- Downstream latency increases and errors amplify
- Retries observed in logs/metrics

## First checks
1. Inspect retry policy (count, backoff, jitter)
2. Check upstream timeout vs downstream p95
3. Confirm whether retries are idempotent

## Mitigation
- Cap retries, add exponential backoff + jitter
- Increase timeout appropriately
- Add circuit breaker
