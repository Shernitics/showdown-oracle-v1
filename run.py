"""
Trains until TARGET_TIMESTEPS (config.py), restarting the Showdown server every CYCLES slices.
"""


import subprocess
import sys
import time
from pathlib import Path

from src.config import SHOWDOWN_DIR


TRAIN = Path(__file__).parent / "src" / "train.py"
SHOWDOWN = Path(SHOWDOWN_DIR)
CYCLES = 8
BOOT = 10
DONE = 3    # exit code from train.py when TARGET_TIMESTEPS is reached


def start_server():
    print("[run] showdown starting", flush=True)
    return subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def stop_server(proc):
    print("[run] showdown stopping", flush=True)
    subprocess.run(
        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)


def main():
    while True:
        server = start_server()
        try:
            time.sleep(BOOT)
            for _ in range(CYCLES):
                code = subprocess.run([sys.executable, str(TRAIN)], cwd=TRAIN.parent).returncode
                if code == DONE:
                    return
                if code:
                    time.sleep(30)
                    break
        finally:
            stop_server(server)


if __name__ == "__main__":
    main()
