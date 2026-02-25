# Runbook: OOM Kill / CrashLoopBackOff

## Symptoms
- Pod restarts, CrashLoopBackOff
- Kernel log shows OOMKilled
- Latency spikes after restart

## First checks
1. Confirm OOMKilled in kubectl describe
2. Check container memory limit vs RSS
3. Identify top allocators / recent traffic spikes

## Mitigation
- Temporarily scale out replicas
- Increase memory limit if safe
- Disable high-memory features / large payload endpoints

## Follow-up
- Add memory profiling and request size guardrails
