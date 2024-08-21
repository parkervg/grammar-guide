import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="speculative_grammar_backtracking",
    python_requires=">=3.9",
    version=find_version("speculative_grammar_backtracking", "__init__.py"),
    url="https://github.com/parkervg/speculative-grammar-backtracking",
    author="Parker Glenn",
    author_email="parkervg5@gmail.com",
    description="LLM decoding with optimistic speculative CFG-guided backtracking.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    # data_files=[
    #     "blendsql/grammars/_cfg_grammar.lark",
    #     "blendsql/prompts/few_shot/hybridqa.txt",
    # ],
    install_requires=["colorama", "lark", "pygtrie", "guidance>=0.1.16", "ipython"],
)
