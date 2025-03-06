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
        "sentence_transformers",
        "sentencepiece",
        "playpen @ git+https://github.com/momentino/playpen",
        "lm_eval @ git+https://github.com/momentino/lm-evaluation-harness",
        "pingouin",
        "langdetect",
        "immutabledict",
    ],
)