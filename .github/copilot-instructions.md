# Copilot Instructions

- Use the workspace virtual environment by `uv` when running code in this repository. The `uv` environment is configured with the appropriate dependencies and settings for this project, and it ensures that your code runs in a consistent environment across different machines and setups.
- The pytest root for this repository is `tests`; prefer targeted test runs under that directory.
- When calling `runTests` with the `files` parameter in this repository, use workspace-relative paths such as `tests/test_import.py`.
- Do not pass absolute filesystem paths to `runTests.files` here. They can return `No tests found in the files` even when test discovery is healthy.
- If a single test must be targeted, `runTests.testNames` works reliably in this repository.