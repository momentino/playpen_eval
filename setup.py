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
        "lm_eval @ git+https://github.com/momentino/lm-evaluation-harness",
        "vllm==0.8.3",
        "pingouin",
        "langdetect",
        "immutabledict",
        "nltk>=3.9.1",
        "flashinfer-python",
    ],
)