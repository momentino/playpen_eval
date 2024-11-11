import argparse
import os
import logging

from pathlib import Path
from playpen_eval.utils import load_yaml_config
from playpen_eval.evaluator.playpen_evaluator import PlaypenEvaluator

playpen_eval_logger = logging.getLogger("playpen_eval_logger")





def main(args: argparse.Namespace) -> None:
    current_folder = Path(os.path.abspath(__file__)).parent
    project_folder = current_folder.parent
    config_file_path = Path(os.path.join(current_folder, "config.yaml"))
    config = load_yaml_config(config_file_path)

    tasks = config.get("tasks", [])
    model = config.get("model", "hf")
    device = config.get("device", "cuda:0")
    trust_remote_code = config.get("trust_remote_code", True)
    log_samples = config.get("log_samples", True)
    output_path = config.get("output_path", "playpen_results")

    if args.command_name == "ls":
        PlaypenEvaluator.list_tasks()
    if args.command_name == "run":
        PlaypenEvaluator.run()
    if args.command_name == "score":
        PlaypenEvaluator.score()

def run(args: argparse.Namespace) -> None:
    current_folder = Path(os.path.abspath(__file__)).parent
    project_folder = current_folder.parent
    config_file_path = Path(os.path.join(current_folder, "config.yaml"))
    config = load_yaml_config(config_file_path)

    tasks = config.get("tasks", [])
    model = config.get("model", "hf")
    device = config.get("device", "cuda:0")
    trust_remote_code = config.get("trust_remote_code", True)
    log_samples = config.get("log_samples", True)
    output_path = config.get("output_path", "playpen_results")

    model_name = args.model_name
    formatted_model_name = model_name.replace("pretrained=", "").replace("/", "__")
    if trust_remote_code:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        model_args = args.model_name + ",trust_remote_code=True"
    else:
        model_args = args.model_name

    output_subfolder = Path(os.path.join(project_folder, output_path)) / formatted_model_name
    output_subfolder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    sub_parsers.add_parser("ls")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)
    run_parser.add_argument(
        "-m", "--model_name",
        type=str,
        required=True,
        help="Model name, e.g., 'pretrained=model-name'."
    )

    score_parser = sub_parsers.add_parser("score", formatter_class=argparse.RawTextHelpFormatter)
    run_parser.add_argument(
        "-m", "--model_name",
        type=str,
        required=True,
        help="Model name, e.g., 'pretrained=model-name'."
    )

    args = parser.parse_args()
    main(args)