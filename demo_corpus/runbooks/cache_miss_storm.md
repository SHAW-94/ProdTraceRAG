# Runbook: Cache Miss Storm / Stampede

## Symptoms
- Cache hit rate drops sharply
- DB QPS spikes; latency increases

## First checks
1. Confirm cache hit rate trend and key space
2. Check TTL / eviction policy changes
3. Identify hot keys and thundering herd

## Mitigation
- Enable request coalescing / singleflight
- Add jitter to TTL
- Pre-warm hot keys
