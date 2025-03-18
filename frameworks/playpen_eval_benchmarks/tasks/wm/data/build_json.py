import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

def merge_data(path: Path) -> List[List[Dict]]:
    trials = []
    sorted_files = sorted(path.iterdir(), key=lambda f: int(f.stem) if f.stem.isdigit() else float('inf'))
    for file_path in sorted_files:
        if file_path.is_file():
            trial = []
            with file_path.open("r", encoding="utf-8") as f:
                seq = f.readline().strip().split(",")
                cond = f.readline().strip().split(",")
                for s,c in zip(seq,cond):
                    trial.append({"stimuli":s, "target":c})
            trials.append(trial)
    return trials

if __name__=="__main__":
    experiments = ["verbal"]
    variants = ["1back","2back","3back"]
    for experiment in experiments:
        for variant in variants:
            output_path_base = Path(__file__).parent / "json" / experiment
            output_path_base.mkdir(parents=True, exist_ok=True)
            output_file_name = f"{variant}.json"
            data_path = Path(__file__).parent / experiment / variant
            variant_dataset = merge_data(data_path)
            json.dump(variant_dataset,open(output_path_base/output_file_name,'w'))





