from setuptools import setup, find_packages
SETUPTOOLS = "setuptools~=45.3.0"

setup(name='sbnet', version='1.0', packages=find_packages(),
    setup_requires=[SETUPTOOLS],
)

