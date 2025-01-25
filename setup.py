from setuptools import setup, find_packages

setup(
    name="playpen_eval",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pyyaml",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "tarski",
        "bitsandbytes",
        "guidance @ git+https://github.com/guidance-ai/guidance.git@hudson-ai-hybridcache", # need to install this branch of guidance repo to work with Gemma2
        "sentence_transformers",
        "sentencepiece",
        "llguidance==0.2.0",
        "playpen @ git+https://github.com/momentino/playpen"

    ],
)