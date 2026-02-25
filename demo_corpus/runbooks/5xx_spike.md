# Runbook: 5xx Rate Spike

## Symptoms
- 5xx rate > threshold, user errors reported
- Dependency latency may increase

## First checks
1. Identify top failing endpoints
2. Check recent deploy/config changes
3. Check downstream error codes and saturation

## Mitigation
- Rollback last change if correlated
- Enable circuit breaker / shed load
- Throttle non-critical traffic
