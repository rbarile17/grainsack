from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="grainsack",
    version="0.1.0",
    description="",
    url="https://github.com/rbarile17/grainsack",
    author="Roberto Barile, Claudia d'Amato, Nicola Fanizzi",
    author_email="r.barile17@phd.uniba.it, claudia.damato@uniba.it, nicola.fanizzi@uniba.it",
    license="MIT",
    packages=find_packages(),
    entry_points={"console_scripts": ["grainsack = grainsack.operations:cli"]},
    install_requires=requirements,
)
