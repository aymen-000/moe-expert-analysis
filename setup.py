from setuptools import setup, find_packages

setup(
    name="moe-expert-analysis",
    version="0.1.0",
    description="Interpretability for sparse Mixture-of-Experts transformers",
    packages=find_packages(include=["moe_interp", "moe_interp.*"]),
    python_requires=">=3.9",
    install_requires=[
        "transformers>=4.40",
        "accelerate",
        "datasets",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
    ],
)
