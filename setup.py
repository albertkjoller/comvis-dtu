from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()


setup(
    name="comvis-dtu",
    version="0.2",
    description="A module gathering functionalities from the course 02504 Comuter Vision @ DTU (Spring 2023)",
    license="MIT",
    long_description=long_description,
    author="Albert Kjøller Jacobsen",
    author_email="s194253@student.dtu.dk",
    url="https://github.com/albertkjoller/comvis-dtu",
    packages=find_packages(),  # ["comvis"],
    install_requires=required_packages,
)
