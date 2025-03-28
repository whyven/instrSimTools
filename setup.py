from setuptools import setup, find_packages

setup(
    name="instrSimTools",  # Name of your package
    version="0.1.0",  # Package version
    author="M. Paniccia",
    author_email="matthewpaniccia@yahoo.com",
    description="Collection of function and classes used to help in the design of instrumentation systems for particle accelerators.",
    long_description=open("README.md").read(),  # Load long description from README.md
    long_description_content_type="text/markdown",
    url="https://github.com/whyven/instrSimTools.git",  # Your GitHub or project page
    packages=find_packages(),  # Automatically find all package modules
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
        "pandas"
    ],  # List dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version required
)
