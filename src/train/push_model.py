from huggingface_hub import HfApi
import argparse
import os


def push_to_hub(model_dir, repo_name):
    api = HfApi()
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory {model_dir} does not exist.")

    api.create_repo(
        repo_id=repo_name,
        token=True,
        private=False,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        repo_type="model",
        token=True,
        commit_message="Pushing model and tokenizer files to hub",
    )
    print(f"Successfully pushed to {repo_name}")


def main(args):
    push_to_hub(model_dir=args.model_dir, repo_name=args.repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--repo_name",
        type=str,
        default="MaxLSB/LeCarnet-3M",
        help="Hugging Face repository name",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/3M",
        help="Directory where the model is saved",
    )
    args = parser.parse_args()
    main(args)
