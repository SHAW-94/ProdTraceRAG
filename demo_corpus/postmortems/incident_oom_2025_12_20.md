# Postmortem: OOM CrashLoop (2025-12-20)

## Impact
- 16:40-17:05 UTC: checkout error rate 8%

## Root Cause
- Large request payload path caused memory blow-up; limit too low

## Fix
- Add request size limit and streaming parsing
- Raise memory limit temporarily; optimize allocation
- Add OOM alert + memory profiling
