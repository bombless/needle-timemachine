from setuptools import find_packages, setup

# Keep explicit legacy metadata so older pip/setuptools versions do not fall
# back to an UNKNOWN package. pyproject.toml remains the canonical metadata.
setup(
    name="needle-timemachine",
    version="0.1.0",
    description="Execution tracing and replay primitives for the Cactus Compute Needle model.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    extras_require={
        "jax": ["jax", "flax"],
        "test": ["pytest", "numpy"],
    },
)
