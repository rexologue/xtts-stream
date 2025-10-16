"""Module with CLI interface for the package."""

from __future__ import annotations

import argparse
from pathlib import Path

from tone import StreamingCTCPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Helper CLI for T-one ASR pipeline", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sub_download = subparsers.add_parser(
        "download",
        help="Download all artifacts from HuggingFace and save them to local folder",
    )
    sub_download.add_argument(
        "download_dir",
        type=Path,
        help="Path to store downloaded artifacts",
    )
    sub_download.add_argument(
        "--only-acoustic",
        action="store_true",
        help="Download only acoustic model (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    """Run main function for CLI."""
    args = parse_args()
    if args.command == "download":
        download_dir: Path = args.download_dir.absolute()
        only_acoustic: bool = args.only_acoustic
        print(f"Downloading all artifacts from HuggingFace to {download_dir}")
        download_dir.mkdir(exist_ok=True)
        StreamingCTCPipeline.download_from_hugging_face(download_dir, only_acoustic=only_acoustic)


if __name__ == "__main__":
    main()
