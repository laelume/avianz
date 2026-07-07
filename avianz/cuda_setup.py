# avianz/cuda_setup.py
# Version 0.1 07/07/26
# Author: laelume

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2026

# Auto-detects driver CUDA support and installs a matching torch build

import re
import subprocess
import sys
import os
import logging

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# C O N F I G
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

VERBOSE = False    # toggle verbose debug logging on/off

# Maps a driver-reported CUDA version to the closest torch wheel index tag.
# nvidia-smi reports the maximum CUDA version the driver supports, not what is
# installed on the system. torch ships its own bundled CUDA runtime, so the
# wheel tag just needs to be less than or equal to the driver's reported max.
# Ordered newest first so the first match found is the newest compatible tag.
#
# cu128 was removed from the torch build matrix as of the 2.11 release line;
# cu130 is now the default/stable tag, with cu126 kept for older drivers.
# This list should be checked against https://pytorch.org/get-started/locally/
# periodically, since torch's supported tag set changes between releases.
CUDA_WHEEL_TAGS = [
    (13, 2, "cu132"),
    (13, 0, "cu130"),
    (12, 6, "cu126"),
]

logger = logging.getLogger("cuda_setup")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)



def get_driver_cuda_version():
    """ Read the maximum CUDA version supported by the installed NVIDIA driver via nvidia-smi.

    Returns a tuple (major, minor) if a driver is found, or None if nvidia-smi
    is not available (no driver, or not an NVIDIA GPU). This reflects driver
    capability only, not whether a CUDA toolkit is installed anywhere.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("nvidia-smi not available or failed: %s", e)
        return None

    # nvidia-smi header line looks like: "... Driver Version: 531.79   CUDA Version: 12.1"
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        logger.debug("nvidia-smi ran but CUDA version not found in output")
        return None

    major, minor = int(match.group(1)), int(match.group(2))
    logger.debug("Driver reports max supported CUDA version: %d.%d", major, minor)
    return major, minor


def pick_wheel_tag(driverVersion, stepDown=False):
    """ Pick a torch wheel tag that the driver's CUDA version supports.

    driverVersion: (major, minor) tuple from get_driver_cuda_version, or None
    stepDown: if True, skips the newest matching tag and picks the next older
              one instead, as a safety margin against a driver being right at
              the edge of support for the newest tag

    Returns a wheel tag string like "cu126", or None if no supported GPU was
    detected, meaning a cpu-only install is the correct choice.
    """
    if driverVersion is None:
        return None

    driverMajor, driverMinor = driverVersion
    matches = [
        tag for tagMajor, tagMinor, tag in CUDA_WHEEL_TAGS
        if (driverMajor, driverMinor) >= (tagMajor, tagMinor)
    ]

    if not matches:
        # FALLBACK: driver detected but older than every known tag in
        # CUDA_WHEEL_TAGS, oldest listed tag is used as the best available match
        oldestTag = CUDA_WHEEL_TAGS[-1][2]
        logger.debug("Driver CUDA %d.%d older than all known tags, using oldest tag %s", driverMajor, driverMinor, oldestTag)
        return oldestTag

    if stepDown and len(matches) > 1:
        selected = matches[1]
        logger.debug("stepDown requested, using %s instead of newest match %s", selected, matches[0])
    else:
        selected = matches[0]
        logger.debug("Selected wheel tag %s for driver CUDA %d.%d", selected, driverMajor, driverMinor)

    return selected


def install_torch(wheelTag):
    """ Install torch, torchvision, and torchaudio using pip, targeting a specific wheel tag or cpu-only.

    wheelTag: string like "cu121" for a CUDA build, or None for cpu-only install
    """
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    if wheelTag is None:
        print("No GPU detected, installing cpu-only torch build")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    else:
        print(f"GPU detected, installing torch build for {wheelTag}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{wheelTag}"
        ])


def auto_install_torch(stepDown=False):
    """ Detect driver CUDA support and install the matching torch build automatically.

    Entry point tying together get_driver_cuda_version, pick_wheel_tag, and
    install_torch. Prints the detected driver version (if any) before installing,
    so the choice made is visible rather than silent.

    stepDown: passed through to pick_wheel_tag, uses the tag one step older
              than the newest match as a safety margin

    Checks AVIANZ_INSTALL_MODE (same variable read by gpu_config.py) before
    running any detection: "cpu" skips detection entirely and installs a
    cpu-only build regardless of hardware, keeping the install choice
    consistent with what configure_gpu_memory will enforce at runtime.
    """
    installMode = os.environ.get("AVIANZ_INSTALL_MODE", "auto")

    if installMode == "cpu":
        # FALLBACK: skip detection entirely, forced cpu-only install to
        # match gpu_config.py's INSTALL_MODE=cpu behaviour at runtime
        print("AVIANZ_INSTALL_MODE=cpu, installing cpu-only torch build")
        install_torch(None)
        return

    driverVersion = get_driver_cuda_version()

    if driverVersion is not None:
        print(f"Detected driver CUDA support: {driverVersion[0]}.{driverVersion[1]}")
    else:
        print("No NVIDIA driver detected via nvidia-smi")

    wheelTag = pick_wheel_tag(driverVersion, stepDown=stepDown)

    if wheelTag is None and installMode == "gpu":
        raise RuntimeError(
            "AVIANZ_INSTALL_MODE=gpu but no NVIDIA driver was detected via "
            "nvidia-smi. GPU mode was explicitly requested, so this is treated "
            "as a setup problem rather than falling back to cpu-only install."
        )

    install_torch(wheelTag)


if __name__ == "__main__":
    auto_install_torch()



# python cuda_setup.py
#
# from cuda_setup import get_driver_cuda_version, pick_wheel_tag, auto_install_torch
#
# driverVersion = get_driver_cuda_version()          # e.g. (13, 0) or None
# wheelTag = pick_wheel_tag(driverVersion)           # newest match, e.g. "cu130"
# wheelTag = pick_wheel_tag(driverVersion, stepDown=True)   # one tag older, e.g. "cu126"
# auto_install_torch()                                # detects and installs newest match
# auto_install_torch(stepDown=True)                    # detects and installs one tag down