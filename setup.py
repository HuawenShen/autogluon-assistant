from setuptools import find_packages, setup

setup(
    name="AutoMLAgent",
    version="0.1.0",
    package_dir={"": "automlagent/src"},  # Updated path
    packages=find_packages(where="automlagent/src"),  # Updated path
    install_requires=[
        "autogluon.assistant",
        "langgraph",
        "autogluon",
        # "speechbrain",
        "FlagEmbedding",
        "faiss-cpu",
    ],
    author="FANGAreNotGnu",
    description="AutoMLAgent beta",
)
