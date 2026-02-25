# Contributing

Thanks for contributing!

## Development Setup

1. Create a Python environment (see `environment.yml` in this repo)
2. Install dependencies
3. Run local API in trusted mode for development
4. Before opening a PR, run:
   - `python -m py_compile app/*.py scripts/*.py eval/*.py`

## Pull Request Guidelines

- Keep changes focused and minimal
- Include tests or a reproducible validation script when possible
- Do not commit secrets or local absolute paths
- Update docs if endpoint behavior or env vars change

## Security-Related Contributions

If your change touches auth, tracing, ingestion, or config endpoints:
- explain the threat model in the PR description
- list secure defaults and opt-out flags
