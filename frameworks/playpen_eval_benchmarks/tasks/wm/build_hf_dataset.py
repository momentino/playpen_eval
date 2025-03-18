# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import datasets
from pathlib import Path


_CITATION = """\
@inproceedings{gong2024working,
  title={Working memory capacity of ChatGPT: An empirical study},
  author={Gong, Dongyu and Wan, Xingchen and Wang, Dingmin},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={9},
  pages={10048--10056},
  year={2024}
}
"""

_DESCRIPTION = """\
A benchmark for evaluating Working Memory capabilities in LLMs. Here only the data for the three base 'verbal' experiments are provided."""

_HOMEPAGE = "https://github.com/Daniel-Gong/ChatGPT-WM"

_LICENSE = "MIT"

_URLS_prefix = {
    "verbal" : "https://raw.githubusercontent.com/momentino/playpen_eval/main/frameworks/playpen_eval_benchmarks/tasks/wm/data/json/verbal",
}
_URLS = {
    "verbal_1back": {
        "test": _URLS_prefix["verbal"] + "/1back.json"
    },
    "verbal_2back": {
        "test": _URLS_prefix["verbal"] + "/2back.json"
    },
    "verbal_3back": {
        "test": _URLS_prefix["verbal"] + "/3back.json"
    }
}

class WorkingMemory(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=config_name,
            version=datasets.Version("0.0.1"),
            description=f"{config_name} task from WorkingMemory"
        )
        for config_name in _URLS.keys()
    ]
    def _info(self):
        features = {
            "stimuli": datasets.Value("string"),
            "target": datasets.Value("string")
        }
        features = datasets.Features(features)
        return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=features,
                homepage=_HOMEPAGE,
                citation=_CITATION,
                license=_LICENSE,
            )

    """def _split_generators(self, dl_manager):
        data_dir = Path("path/to/your/local/folder")  # Use Path object
        subset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]  # Get only directories

        split_generators = []
        for subset_dir in subset_dirs:
            for i in range(50):  # Create at least 50 splits per subset
                split_generators.append(
                    datasets.SplitGenerator(
                        name=f"{subset_dir.name}_split_{i}",
                        gen_kwargs={
                            "filepath": str(subset_dir),
                            "split": f"{subset_dir.name}_split_{i}",
                        },
                    )
                )

        return split_generators"""

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        with open(data_dir["test"], encoding="utf-8") as fin:
            data = json.load(fin)

        # Create one split per instance, naming them uniquely
        splits = []
        for idx in range(len(data)):
            splits.append(
                datasets.SplitGenerator(
                    # Name splits as "test_0", "test_1", etc.
                    name=f"{idx}",
                    gen_kwargs={
                        "filepath": data_dir["test"],
                        "index": idx,
                    }
                )
            )
        return splits


    def _generate_examples(self, filepath, index):
        # Open the JSON file and load the instance at the provided index
        with open(filepath, encoding="utf-8") as fin:
            data = json.load(fin)
            for id,instance in enumerate(data[index]):
                # Yield using the instance id as key (make sure it's unique)
                yield id, instance