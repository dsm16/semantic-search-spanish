"""Anonymous, pinned download of three gold files; standard library only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from .dataset import load_corpus

REPOSITORY = "DReggio/esg-spanish-events"
REVISION = "e900148694757d37df41035df090dd640052874d"
EXPECTED_SHA256 = {
    "gold_train.csv": "ef529adcae0a1c2138d249d35a61e58b57f8ecdc793778567f31947fed43a256",
    "gold_dev.csv": "1ccc62efe4cee7a2ff1f13a49cb4868f8473050960a8c03ca99f859ae901d0b5",
    "gold_test.csv": "cdbfeb7c22b1913d5b0047e17bfd6b7ce54d1b484bda6e47cb690f7a932096ee",
}
DEFAULT_DIRECTORY = "data/raw/esg-spanish-events"


def verify_file(path: str | Path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch: {Path(path).name}")


def download_gold(directory: str | Path = DEFAULT_DIRECTORY) -> tuple[Path, ...]:
    """Refuse archival destinations and existing mismatches; verify before rename.

    Existing valid files are reused. Each file is independently atomic; a failed
    later download may leave earlier verified files for a subsequent retry.
    """
    directory = Path(directory).resolve()
    if any(part.casefold() == "_originals" for part in directory.parts):
        raise ValueError("Downloads must never target _originals")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in EXPECTED_SHA256.items():
        target = directory / name
        if target.exists():
            verify_file(target, expected)
        else:
            url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
            temporary = None
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                        temporary = Path(handle.name)
                        while chunk := response.read(65536):
                            handle.write(chunk)
                verify_file(temporary, expected)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        paths.append(target)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY)
    parser.add_argument("--stats-only", action="store_true",
                        help="Read existing gold files without network access")
    args = parser.parse_args()
    if not args.stats_only:
        download_gold(args.directory)
    print(json.dumps(load_corpus(args.directory).statistics, indent=2))


if __name__ == "__main__":
    main()
