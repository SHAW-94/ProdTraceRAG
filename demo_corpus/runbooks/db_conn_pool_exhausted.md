# Runbook: DB Connection Pool Exhausted

## Symptoms
- DB errors: too many connections / pool timeout
- API latency increases; 5xx may rise

## First checks
1. Check active connections / max_connections
2. Verify pool size settings and leaks
3. Inspect slow queries and lock waits

## Mitigation
- Reduce concurrency / add rate limit
- Increase pool size cautiously
- Kill runaway queries; add timeouts
