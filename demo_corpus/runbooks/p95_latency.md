# Runbook: P95 Latency Spike

## Symptoms
- P95 latency increases rapidly
- Error rate may remain stable initially

## First checks
1. Confirm time window and affected regions
2. Check dependency status (DB, cache, downstream services)
3. Inspect CPU/memory saturation

## Mitigation
- Apply rate limiting for non-critical endpoints
- Roll back recent deployment if correlated
- Increase replicas if safe
