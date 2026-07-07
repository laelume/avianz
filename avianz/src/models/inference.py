# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean
#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# PyTorch inference utilities

import torch


from avianz.src.core import spectrogram
from avianz.src.core import config_loader
from avianz.src.core import audio_data
from avianz.src.utils import shapes
from avianz.src.models import inference

import numpy as np


def configure_gpu_memory(model=None, verbose=None):
    """ Select and configure inference device, moving model onto it if provided.

    Preserves cache-clearing behaviour when called with no model (existing call
    sites keep working unchanged), and additionally returns the resolved torch.device so
    callers can track where inference is actually executing. Also moves model itself onto
    that device, this was previously missing, which meant GPU was never actually engaged
    even when available.

    When no GPU is detected, prompts at the command line for y/n confirmation before
    proceeding on CPU, rather than silently falling back.
    """
    if verbose is not None:
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.debug("GPU detected, running inference on cuda device")
    else:
        # FALLBACK: cpu execution path, engaged whenever cuda is unavailable.
        # Requires explicit y/n confirmation at the command line before proceeding,
        # rather than silently falling back to cpu.
        response = input("No GPU detected. Proceed with CPU? [y/n]: ").strip().lower()
        while response not in ("y", "n"):
            response = input("Please enter 'y' or 'n': ").strip().lower()

        if response == "n":
            logger.debug("User declined cpu fallback, aborting")
            raise RuntimeError("No GPU detected and CPU fallback declined")

        device = torch.device("cpu")
        print("Proceeding with CPU")
        logger.debug("Proceeding with cpu device after user confirmation")
        configure_cpu_threads()

    if model is not None:
        model.to(device)
        return model, device

    return device
    
# PyTorch inference utilities

import os
import logging
import torch
from concurrent.futures import ThreadPoolExecutor

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# C O N F I G
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

VERBOSE = False            # toggle verbose debug logging on/off
CPU_THREAD_COUNT = None    # None -> fall back to os.cpu_count() at runtime

logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)


def configure_gpu_memory(model=None, verbose=None):
    """ Select and configure inference device (GPU if available, else CPU), and move model onto it.

    Preserves original cache-clearing behaviour when called with no model (existing call
    sites keep working unchanged), and now additionally returns the resolved torch.device so
    callers can track where inference is actually executing, plus moves the model itself onto
    that device -- this was previously missing, which meant GPU was never actually engaged
    even when available.
    """
    if verbose is not None:
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.debug("GPU detected, running inference on cuda device")
    else:
        device = torch.device("cpu")
        logger.debug("No GPU detected, falling back to cpu device")
        # FALLBACK: cpu execution path used whenever cuda is unavailable
        configure_cpu_threads()

    if model is not None:
        model.to(device)
        return model, device

    return device


def configure_cpu_threads(num_threads=None):
    """ Set the number of CPU threads torch uses for intra-op parallelism.

    Applies on the cpu-only fallback path, and also alongside GPU inference for any
    CPU-bound pre/post-processing torch ops that run outside the model forward pass.
    Falls back to os.cpu_count() if num_threads is not given and CPU_THREAD_COUNT is unset.
    """
    threads = num_threads or CPU_THREAD_COUNT or os.cpu_count() or 1
    torch.set_num_threads(threads)
    logger.debug("torch cpu thread count set to %d", threads)


def generate_features_threaded(segment_args, feature_fn, max_workers=None):
    """ Generate NN input features for multiple segments concurrently via a thread pool.

    segment_args: ordered list of argument-tuples, one per segment, to be unpacked into feature_fn
    feature_fn: callable performing the CPU-bound feature generation for a single segment
                (e.g. PostProcess.generate_nn_features)
    max_workers: worker count; falls back to CPU_THREAD_COUNT or os.cpu_count()

    Order of results matches segment_args (uses pool.map rather than as-completed), which
    keeps downstream indexing against the caller's segment list aligned.
    """
    workers = max_workers or CPU_THREAD_COUNT or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda args: feature_fn(*args), segment_args))
    return results


def predict_batch(model, features):
    """ Run batch prediction using the model. """
    model.eval()
    with torch.no_grad():
        if isinstance(features, torch.Tensor):
            tensor_input = features
        else:
            tensor_input = torch.from_numpy(features).float()

        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if len(tensor_input.shape) == 4 and tensor_input.shape[-1] == 1:
            tensor_input = tensor_input.permute(0, 3, 1, 2)
        elif len(tensor_input.shape) == 3:
            tensor_input = tensor_input.unsqueeze(1)

        device = next(model.parameters()).device
        tensor_input = tensor_input.to(device)

        predictions = model(tensor_input)
        return predictions.cpu().numpy()
