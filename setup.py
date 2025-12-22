from setuptools import setup, find_packages

setup(
    name="ego2exo_bench",
    version="0.1.0",
    packages=find_packages(include=['ego2exo_bench*', 'configs']),
    package_dir={'': '.'},
)