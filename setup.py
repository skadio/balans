import os

import setuptools
from setuptools.command.install import install as _install

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

with open(os.path.join('xxx', '_version.py')) as f:
    exec(f.read())


class Install(_install):
    def run(self):
        # _install.do_egg_install(self)
        _install.run(self)
        import nltk
        nltk.download("punkt")


# python setup.py bdist_wheel
setuptools.setup(
    name="xxx",
    description="xxx",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author="xxx",
    url="https://xxx",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "notebooks"]),
    cmdclass={"install": Install},
    install_requires=required,
    setup_requires=["nltk"],
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://xxx",
        "Source": "https://xxx"
    }
)