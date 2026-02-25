# Postmortem: Cache Stampede (2025-12-09)

## Impact
- 14:01-14:18 UTC: DB QPS x5; p95 latency +300ms

## Root Cause
- TTL change caused synchronized expiration on hot keys (no jitter)

## Fix
- Add TTL jitter and singleflight
- Pre-warm hot keys on deploy
- Add cache hit rate SLO alerts
