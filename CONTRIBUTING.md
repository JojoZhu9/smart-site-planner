# Contributing

Thank you for improving Smart Site Planner. This is collaborative course/project work; preserve existing team and course attribution and do not represent the project as sole authorship.

## Before You Start

- Discuss substantial algorithm or workflow changes in an issue before implementation.
- Do not commit private POI, location, customer, operational, credential, runtime-output, or local-environment data.
- Keep changes focused and update English documentation first; add concise Chinese updates in `README_ZH.md` when user-facing behavior changes.

## Development Checklist

1. Install dependencies with `pip install -r requirements.txt`.
2. Use a reproducible sample dataset that has the documented schema and contains no private location data.
3. Run `python -m compileall main.py src test_main.py` before submitting.
4. Add or update automated tests for every algorithm behavior change. The repository does not currently include collected pytest tests, so new tests must be independently runnable and documented.
5. Include before-and-after screenshots for Streamlit or visualization changes. Do not include sensitive map labels, coordinates, or location data in screenshots.
6. Describe configuration, schema, or output changes in both README files.

## Pull Requests

Use the pull request template. Explain the problem, the approach, validation performed, sample-data handling, and any known limitations. Keep generated outputs, caches, IDE files, and logs out of the pull request.

## Reporting Problems

Use the issue templates for public bugs and feature requests. Send security-sensitive reports through the process in [SECURITY.md](SECURITY.md).
