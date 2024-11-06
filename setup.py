# setup.py
from setuptools import setup, find_packages

setup(
    name="parallelgsp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numba>=0.60.0",
    ],
    author="dxZhang",
    author_email="dxzhang49@outlook.com",
    description="A Generalized Sequential Pattern Mining Algorithm Implementation with Numba",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danxuZhang/parallelGSP",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

