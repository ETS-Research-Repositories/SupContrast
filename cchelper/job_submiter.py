import os
import random
import string
import subprocess
from pprint import pprint
from typing import List, Union


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


def sbatch_script_prefix(
    account,
    time=1,
    job_name="default_jobname",
    nodes=1,
    gres="gpu:1",
    cpus_per_task=6,
    mem=16,
    mail_user="jizong.peng.1@etsmtl.net",
    *args,
    **kwargs,
):
    sbatch_basis = (
        f"#!/bin/bash \n"
        f"#SBATCH --time=0-{time}:00 \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --cpus-per-task={cpus_per_task} \n"
        f"#SBATCH --gres={gres} \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --job-name={job_name} \n"
        f"#SBATCH --nodes={nodes} \n"
        f"#SBATCH --mem={mem}000M \n"
        f"#SBATCH --mail-user={mail_user} \n"
        f"#SBATCH --mail-type=ALL \n"
    )
    return sbatch_basis


class JobSubmiter:
    def __init__(self, project_path="./", on_local=False, **kwargs) -> None:
        self._project_path = project_path
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._on_local = on_local

    def prepare_env(self, exec: Union[str, List[str]] = ""):
        if isinstance(exec, str):
            exec = [
                exec,
            ]
        self.exec_env = exec

    def run(self, job_script: str):
        pprint(job_script)
        sbatch_script = sbatch_script_prefix(**{k: v for k, v in self.__dict__.items()})
        env_script = "\n".join(self.exec_env) if hasattr(self, "exec_env") else ""
        full_script = "\n".join([sbatch_script, env_script, job_script])
        self._write_and_run(full_script)

    def _write_and_run(self, full_script):
        random_name = randomString() + ".sh"
        file_fullpath = os.path.join(self._project_path, random_name)
        with open(file_fullpath, "w") as f:
            f.write(full_script)
        try:
            if self._on_local:
                subprocess.call(f"bash {file_fullpath}", shell=True)
            else:
                subprocess.call(f"sbatch {file_fullpath}", shell=True)
        finally:
            os.remove(file_fullpath)


if __name__ == "__main__":
    job_sumbmitter = JobSubmiter()
    job_sumbmitter.account = "def-chdesa"
    job_sumbmitter.prepare_env("source activate")
    job_sumbmitter.run("nvidia-smi")
