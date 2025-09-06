from setuptools import setup

# Load dependencies
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Load project README for long description
with open("README.md", encoding="utf-8") as f:
    readme_text = f.read()

setup(
    name="incentivus-torch",
    version="1.0.0",
    description="Incentivus: Scalable Training with Efficient Optimization",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/incentivus/Incentivus",
    author="The Incentivus Contributors",
    author_email="team@incentivus.org",
    license="Apache-2.0",
    packages=["incentivus_torch"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
