from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
import modal

APP_NAME = "astrobridge-evals"
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.57",
        "peft>=0.11",
        "accelerate>=0.30",
        "huggingface_hub>=0.23",
        "pandas>=2.2",
        "pyarrow>=15.0",
        "numpy>=1.26",
        "omegaconf>=2.3",
        "astropy>=6.0",
        "tqdm>=4.66",
        "pillow>=10.0",
        "torchvision>=0.17",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.4.0",
        "google-genai>=2.22.0",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/root/astrobridge-eval")
)


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=86400,
    secrets=[
        modal.Secret.from_name("gemini-api-key", required=False),
        modal.Secret.from_name("huggingface-secret", required=False),
    ],
)
def run_command_remote(script: str, args: str) -> None:
    workdir = "/root/astrobridge-eval"
    cmd = [sys.executable, script] + shlex.split(args)
    print(f"Running on Modal remote: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workdir}/src:{workdir}"
    subprocess.run(cmd, cwd=workdir, env=env, check=True)


@app.local_entrypoint()
def main(
    script: str = "eval_scripts/run_caption_eval.py",
    args: str = "--task eval_configs/tasks/distance.yaml --responder eval_configs/responders/astrobridge.yaml --frontier eval_configs/frontier/gemini.yaml",
) -> None:
    print(f"Dispatching script '{script}' to Modal with args: {args}")
    run_command_remote.remote(script=script, args=args)

