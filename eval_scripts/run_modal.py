from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import modal

APP_NAME = "astrobridge-evals"
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)


def download_qwen() -> None:
    import os
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen3.5-9B"
    print(f"Pre-downloading base model weights: {model_id}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    snapshot_download(repo_id=model_id, token=token)


eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_pyproject(str(REPO_ROOT / "pyproject.toml"))
    .run_function(download_qwen)
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/astrobridge-eval/src")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/astrobridge-eval/configs")
    .add_local_dir(str(REPO_ROOT / "eval_configs"), remote_path="/root/astrobridge-eval/eval_configs")
    .add_local_dir(str(REPO_ROOT / "eval_scripts"), remote_path="/root/astrobridge-eval/eval_scripts")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/astrobridge-eval/scripts")
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/astrobridge-eval/pyproject.toml")
)


MODAL_GPU = os.environ.get("MODAL_GPU", "A100-80GB")


@app.function(
    image=eval_image,
    gpu=MODAL_GPU,
    timeout=86400,
)
def run_command_remote(
    script: str, args: str, output_path: str | None = None
) -> bytes | None:
    workdir = "/root/astrobridge-eval"
    python_bin = shutil.which("python") or sys.executable
    cmd = [python_bin, script] + shlex.split(args)
    print(f"Running on Modal remote: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workdir}/src:{workdir}"
    subprocess.run(cmd, cwd=workdir, env=env, check=True)

    if output_path:
        target = Path(workdir) / output_path
        if target.is_file():
            print(f"Reading output file {target} to download locally...")
            return target.read_bytes()
        elif target.is_dir():
            print(f"Archiving output directory {target} to download locally...")
            import io
            import tarfile

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(target), arcname=target.name)
            return buf.getvalue()
    return None


@app.local_entrypoint()
def main(
    script: str = "eval_scripts/generate_captions.py",
    args: str = "--responder eval_configs/responders/astrobridge.yaml --output eval_results/cached_captions/astrobridge_captions.jsonl --benchmark all",
    output: str | None = None,
) -> None:
    target_output = output
    if not target_output:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--output-dir", type=str, default=None)
        parsed, _ = parser.parse_known_args(shlex.split(args))
        target_output = parsed.output or parsed.output_dir

    print(f"Dispatching script '{script}' to Modal with args: {args}")
    result_bytes = run_command_remote.remote(
        script=script, args=args, output_path=target_output
    )

    if result_bytes and target_output:
        dest = Path(target_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if result_bytes[:2] == b"\x1f\x8b":  # gzip compressed archive
            import io
            import tarfile

            try:
                with tarfile.open(fileobj=io.BytesIO(result_bytes), mode="r:gz") as tar:
                    tar.extractall(dest.parent)
                print(
                    f"Successfully downloaded and extracted output directory to {dest}"
                )
                return
            except Exception:
                pass
        dest.write_bytes(result_bytes)
        print(
            f"Successfully downloaded output file to {dest} ({len(result_bytes)} bytes)"
        )
