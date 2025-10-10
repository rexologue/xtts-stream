import argparse
import shutil
import os
from huggingface_hub import snapshot_download

REPO_ID = "diglab-tts/ruxtts"

def download_repo(local_dir: str, token: str = ""):
    # Скачиваем весь репозиторий
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        cache_dir="hf_temp_cache",
        token=token
    )

    # Удаляем внутренний .cache, если образовался
    cache_inside = os.path.join(local_dir, ".cache")
    if os.path.isdir(cache_inside):
        shutil.rmtree(cache_inside)

def main():
    parser = argparse.ArgumentParser(description="Download full XTTS2 repo into specified folder")
    parser.add_argument("out", type=str, help="Output local directory")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (for private repos)")
    args = parser.parse_args()

    # Создаём папку, если её нет
    os.makedirs(args.out, exist_ok=True)

    download_repo(args.out, token=args.token)

if __name__ == "__main__":
    main()
