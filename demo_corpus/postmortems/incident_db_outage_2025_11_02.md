# Postmortem: DB Outage (2025-11-02)

## Impact
- 09:10-09:32 UTC: order write failures peaked at 22%
- Elevated latency across read paths

## Root Cause
- Primary DB node failed; failover delayed by misconfigured health checks

## Fix
- Correct health check thresholds
- Add read-only degradation mode
- Improve failover runbook and alerting
