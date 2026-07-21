# Security Policy

## Sensitive Data

POI and location datasets can reveal operational locations, coverage patterns, and customer or business information. Do not commit raw datasets, production outputs, screenshots containing sensitive coordinates, or derived location data unless they are explicitly authorized and safely anonymized.

## Credentials and Local Configuration

Do not commit API keys, tokens, passwords, connection strings, environment files, runtime databases, or machine-specific paths. If a credential is exposed, revoke or rotate it through the relevant provider before reporting the incident.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository maintainers rather than opening a public issue. Include a concise description, affected files or dependencies, reproduction steps that use non-sensitive data, impact, and any mitigation already taken. Do not attach private datasets, credentials, or production logs.

## Dependency Issues

For a suspected vulnerable dependency, include the package name, installed version, advisory or CVE link when available, and the affected command or workflow. Avoid publishing exploit details until maintainers have had a reasonable opportunity to assess and address the report.
