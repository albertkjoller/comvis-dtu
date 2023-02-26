from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="comvis-dtu",
    version="0.1",
    description="A module gathering functionalities from the course 02504 Comuter Vision @ DTU (Spring 2023)",
    license="MIT",
    long_description=long_description,
    author="Albert Kjøller Jacobsen",
    author_email="s194253@student.dtu.dk",
    url="https://github.com/albertkjoller/comvis-dtu",
    packages=["comvis"],
    install_requires=required_packages,
)
