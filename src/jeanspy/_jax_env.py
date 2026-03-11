from __future__ import annotations

import os


def configure_jax_environment() -> None:
    """Apply process-wide JAX environment defaults before importing JAX."""
    requested_platform = os.environ.get("JEANSPY_JAX_PLATFORM", "").strip().lower()
    if requested_platform:
        mapped_platform = "cuda" if requested_platform == "gpu" else requested_platform
        os.environ.setdefault("JAX_PLATFORMS", mapped_platform)

    legacy_x64 = os.environ.get("JEANSPY_JAX_ENABLE_X64")
    if legacy_x64 is not None:
        os.environ.setdefault("JAX_ENABLE_X64", legacy_x64)

    os.environ.setdefault("JAX_ENABLE_X64", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")