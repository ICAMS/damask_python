"""Miscellaneous helper functionality."""

import sys as _sys
import datetime as _datetime
import os as _os
import subprocess as _subprocess
import shlex as _shlex
import re as _re
import signal as _signal
import fractions as _fractions
import contextlib as _contextlib
import numbers as _numbers
from collections import abc as _abc
from functools import reduce as _reduce, partial as _partial
from pathlib import Path as _Path
import logging
import warnings as _warnings
from typing import Optional as _Optional, Union as _Union, Iterable as _Iterable, \
                   Literal as _Literal, NamedTuple as _NamedTuple, \
                   Any as _Any, TextIO as _TextIO, Generator as _Generator, Sequence as _Sequence

import math as _math
import numpy as _np
import h5py as _h5py

from . import version as _version
from ._typehints import FloatSequence as _FloatSequence, IntSequence as _IntSequence, \
                        NumpyRngSeed as _NumpyRngSeed, FileHandle as _FileHandle

class stdioTuple(_NamedTuple):
    stdout: str
    stderr: str


logger = logging.getLogger(__name__)

# https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
# https://stackoverflow.com/questions/287871
_colors = {
           'header' :   '\033[95m',
           'OK_blue':   '\033[94m',
           'OK_green':  '\033[92m',
           'warning':   '\033[93m',
           'fail':      '\033[91m',
           'end_color': '\033[0m',
           'bold':      '\033[1m',
           'dim':       '\033[2m',
           'underline': '\033[4m',
           'crossout':  '\033[9m'
          }

####################################################################################################
# Functions
####################################################################################################
def _build_loadsteps_from_mechanical_bc(
    data: dict[str, _Any],
    f_out_list: _Optional[_Sequence[int]] = None,
    f_restart_list: _Optional[_Sequence[int]] = None,
) -> list[dict[str, _Any]]:
    """
    Build DAMASK grid/FFT loadsteps from the updated MiMeDat mechanical_BC format.

    Supported loading_type values:
        strain, stress, strain_rate, stress_rate,
        strain-stress, strain_rate-stress, stress-strain_rate.

    Supported loading_mode values:
        static, monotonic, cyclic.

    MiMeDat-to-DAMASK mapping:
        strain              -> F + complementary P
        stress              -> P + complementary dot_F
        strain_rate         -> dot_F + complementary P
        stress_rate         -> dot_P + complementary dot_F
        strain-stress       -> F + P
        strain_rate-stress  -> dot_F + P
        stress-strain_rate  -> P + dot_F

    Scientific conventions:
        - strain is engineering strain and is converted to F:
              F_ii = 1 + strain_ii, F_ij = strain_ij for i != j
        - strain_rate is converted to dot_F directly:
              dot_F_ij = strain_rate_ij
          No +1 is added because d(I)/dt = 0.
        - stress must already be first Piola-Kirchhoff stress in Pa.
        - stress_rate must already be first Piola-Kirchhoff stress rate in Pa/s.
        - No stress-unit conversion is performed.

    Cyclic convention:
        DAMASK has no cyclic load keyword, so a MiMeDat cyclic block is expanded into
        ordinary DAMASK loadsteps. Cyclic loading is supported only for absolute
        quantities: strain, stress, and the mixed strain-stress type. Rate-controlled
        cyclic loading (strain_rate, stress_rate, and mixed types containing a rate) is
        rejected, because this convention is defined on absolute extrema, not rates.

        One full cycle is a closed, fully reversed loop written as four equal
        quarter-cycle loadsteps:

            0 -> max -> 0 -> min -> 0      (target factors 1 -> 0 -> R -> 0)

        with max = magnitude and min = R * magnitude. R < 0 is required, because the
        loop is only meaningful for reversed (sign-changing) loading. For strain this
        maps to F via F_ii = 1 + strain_ii; e.g. strain_max = 0.1, R = -1 gives the
        F_xx target sequence 1.1 -> 1.0 -> 0.9 -> 1.0.

        Timing:
            n_cycles     = duration * frequency        (must be a positive integer)
            segment_time = 1 / (4 * frequency)         (per quarter-cycle loadstep)
            N            = segment_time / step         (must be a positive integer)
        The four-segment block is repeated n_cycles times, so the total generated time
        equals 'duration'. 'duration' is the total cyclic duration, 'step' is the
        physical time increment, and DAMASK increments are N = t / step.

    Tensor convention:
        Required keys: xx, yy, zz, xy, xz, yz.
        Optional lower-triangle keys: yx, zx, zy.
        Off-diagonal input is interpreted as the deformation-gradient / displacement-
        gradient component F_ij (i != j), i.e. the *tensor* shear, NOT the engineering
        shear angle gamma_ij = 2*eps_ij. A symmetric input (xy provided, yx mirrored)
        builds F = I + eps with F_ij = F_ji = eps_ij; the corresponding engineering
        shear is 2*eps_ij.
        Missing lower-triangle keys are mirrored from the corresponding upper term.
        If a missing lower term is mirrored from a nonzero value, a warning is emitted
        because this creates symmetric shear. For simple shear, provide the lower term
        explicitly, e.g. xy=gamma and yx=0.0.

    Stress convention:
        stress and stress_rate are the first Piola-Kirchhoff stress (P) and its rate,
        in Pa and Pa/s. No conversion from Cauchy/nominal stress is performed; the caller
        must supply PK1. (At small strain PK1, nominal, and Cauchy coincide; at finite
        strain they do not.)
    """

    supported_loading_types = {
        "strain", "stress", "strain_rate", "stress_rate",
        "strain-stress", "strain_rate-stress", "stress-strain_rate",
    }
    supported_loading_modes = {"static", "monotonic", "cyclic"}

    tensor_keys = [["xx", "xy", "xz"], ["yx", "yy", "yz"], ["zx", "zy", "zz"]]
    required_keys = {"xx", "yy", "zz", "xy", "xz", "yz"}
    mirror_map = {"yx": "xy", "zx": "xz", "zy": "yz"}

    def is_number(value: _Any) -> bool:
        return isinstance(value, _numbers.Real) and not isinstance(value, bool)

    def clean_float(value: _Any) -> float:
        # Remove only negative-zero representation; do not clip small values.
        result = float(value)
        return 0.0 if result == 0.0 else result

    def is_x(value: _Any) -> bool:
        return isinstance(value, str) and value == "x"

    def terms(loading_type: str) -> list[str]:
        return loading_type.split("-")

    def uses_stress(loading_type: str) -> bool:
        return any(term in {"stress", "stress_rate"} for term in terms(loading_type))

    def uses_rate(loading_type: str) -> bool:
        return any(term in {"strain_rate", "stress_rate"} for term in terms(loading_type))

    def validate_units(loading_type: str) -> None:
        if uses_stress(loading_type):
            units = data.get("units")
            if not isinstance(units, dict):
                raise ValueError(
                    "Stress/stress_rate loading requires data['units']['Stress'] == 'Pa'. "
                    "No stress-unit conversion is performed."
                )
            if units.get("Stress") != "Pa":
                raise ValueError(
                    "Stress/stress_rate values must already be in Pa or Pa/s. "
                    f"Expected data['units']['Stress'] == 'Pa', got {units.get('Stress')!r}."
                )

        units = data.get("units")
        if isinstance(units, dict) and "Time" in units and units["Time"] != "s":
            raise ValueError(
                "This converter assumes duration, step, and frequency are in seconds. "
                f"Expected data['units']['Time'] == 's', got {units['Time']!r}."
            )

    def validate_scalar(key: str, value: _Any, allow_x: bool, context: str) -> None:
        if is_number(value):
            return
        if allow_x and is_x(value):
            return
        expected = "a number or lowercase 'x'" if allow_x else "a number"
        raise ValueError(f"{context}: invalid value for {key!r}: {value!r}. Expected {expected}.")

    def compute_N(duration: float, step: float, context: str) -> int:
        if duration <= 0.0:
            raise ValueError(f"{context}: duration must be > 0, got {duration}.")
        if step <= 0.0:
            raise ValueError(f"{context}: step must be > 0, got {step}.")
        n_float = duration / step
        n_int = int(round(n_float))
        if not _math.isclose(n_float, n_int, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(
                f"{context}: duration / step must be an integer. "
                f"Got duration={duration}, step={step}, duration/step={n_float}."
            )
        if n_int < 1:
            raise ValueError(f"{context}: computed N must be >= 1, got {n_int}.")
        return n_int

    def read_tensor(magnitude: dict[str, _Any], allow_x: bool, context: str) -> list[list[_Union[float, str]]]:
        if not isinstance(magnitude, dict):
            raise TypeError(f"{context}: magnitude tensor must be a dictionary.")

        missing = sorted(required_keys - set(magnitude.keys()))
        if missing:
            raise ValueError(f"{context}: missing required components: {missing}.")

        components = dict(magnitude)
        for lower_key, upper_key in mirror_map.items():
            if lower_key not in components:
                upper_value = components[upper_key]
                components[lower_key] = upper_value
                if is_number(upper_value) and abs(float(upper_value)) > 0.0:
                    _warnings.warn(
                        f"{context}: {lower_key!r} was missing and mirrored from nonzero "
                        f"{upper_key!r}={upper_value!r}. This creates symmetric shear. "
                        f"For simple shear, provide {lower_key!r} explicitly, e.g. {lower_key}=0.0.",
                        UserWarning,
                        stacklevel=2,
                    )

        for row_keys in tensor_keys:
            for key in row_keys:
                validate_scalar(key, components[key], allow_x, context)

        tensor: list[list[_Union[float, str]]] = []
        for row_keys in tensor_keys:
            row: list[_Union[float, str]] = []
            for key in row_keys:
                value = components[key]
                row.append("x" if is_x(value) else clean_float(value))
            tensor.append(row)
        return tensor

    def scale_magnitude_dict(magnitude: dict[str, _Any], factor: float, context: str) -> dict[str, _Any]:
        if not isinstance(magnitude, dict):
            raise TypeError(f"{context}: magnitude tensor must be a dictionary.")
        scaled: dict[str, _Any] = {}
        for key, value in magnitude.items():
            if is_x(value):
                scaled[key] = "x"
            elif is_number(value):
                scaled[key] = clean_float(float(value) * factor)
            else:
                raise ValueError(f"{context}: invalid value for {key!r}: {value!r}. Expected number or 'x'.")
        return scaled

    def scale_magnitude_object(magnitude: _Any, factor: float, context: str) -> _Any:
        if isinstance(magnitude, dict):
            return scale_magnitude_dict(magnitude, factor, context)
        if isinstance(magnitude, list):
            return [scale_magnitude_dict(t, factor, f"{context}, tensor[{i}]") for i, t in enumerate(magnitude)]
        raise TypeError(f"{context}: magnitude must be a tensor dict or two-tensor list.")

    def build_F_from_strain(tensor: list[list[_Union[float, str]]]) -> list[list[_Union[float, str]]]:
        F: list[list[_Union[float, str]]] = []
        for i, row in enumerate(tensor):
            F_row: list[_Union[float, str]] = []
            for j, value in enumerate(row):
                if is_x(value):
                    F_row.append("x")
                elif i == j:
                    F_row.append(clean_float(1.0 + float(value)))
                else:
                    F_row.append(clean_float(value))
            F.append(F_row)
        return F

    def build_dot_F_from_strain_rate(tensor: list[list[_Union[float, str]]]) -> list[list[_Union[float, str]]]:
        dot_F: list[list[_Union[float, str]]] = []
        for row in tensor:
            dot_F.append(["x" if is_x(value) else clean_float(value) for value in row])
        return dot_F

    def damask_key(quantity: str) -> str:
        if quantity == "strain":
            return "F"
        if quantity == "strain_rate":
            return "dot_F"
        if quantity == "stress":
            return "P"
        if quantity == "stress_rate":
            return "dot_P"
        raise ValueError(f"Unsupported tensor quantity: {quantity!r}.")

    def map_to_damask_tensor(tensor: list[list[_Union[float, str]]], quantity: str) -> list[list[_Union[float, str]]]:
        if quantity == "strain":
            return build_F_from_strain(tensor)
        if quantity == "strain_rate":
            return build_dot_F_from_strain_rate(tensor)
        if quantity in {"stress", "stress_rate"}:
            return tensor
        raise ValueError(f"Unsupported tensor quantity: {quantity!r}.")

    def complement_for_pure(key: str, tensor: list[list[_Union[float, str]]]) -> dict[str, list[list[_Union[float, str]]]]:
        complement: list[list[_Union[float, str]]] = []
        for row in tensor:
            complement.append([0.0 if is_x(value) else "x" for value in row])
        if key in {"F", "dot_F"}:
            return {"P": complement}
        if key in {"P", "dot_P"}:
            return {"dot_F": complement}
        raise ValueError(f"Unsupported DAMASK key for complement: {key!r}.")

    def validate_mixed_complementarity(
        tensor_a: list[list[_Union[float, str]]],
        tensor_b: list[list[_Union[float, str]]],
        context: str,
    ) -> None:
        for i, row_keys in enumerate(tensor_keys):
            for j, key in enumerate(row_keys):
                a_is_x = is_x(tensor_a[i][j])
                b_is_x = is_x(tensor_b[i][j])
                if a_is_x == b_is_x:
                    raise ValueError(
                        f"{context}: mixed loading is not complementary at component {key!r}. "
                        "Exactly one tensor must contain a number and the other must contain 'x'."
                    )

    def build_mechanical_bc(loading_type: str, magnitude: _Any, context: str) -> dict[str, list[list[_Union[float, str]]]]:
        if "-" not in loading_type:
            tensor = read_tensor(magnitude, allow_x=False, context=context)
            key = damask_key(loading_type)
            damask_tensor = map_to_damask_tensor(tensor, loading_type)
            mechanical_bc = {key: damask_tensor}
            mechanical_bc.update(complement_for_pure(key, damask_tensor))
            if loading_type in {"stress", "stress_rate"}:
                _warnings.warn(
                    f"{context}: pure {loading_type!r} loading prescribes the full stress tensor and "
                    "leaves every deformation component free (a fully stress-controlled boundary "
                    "condition). This is valid but tends to be numerically less robust in the spectral "
                    "solver than a mixed condition. For a uniaxial or otherwise simple stress state, "
                    "prefer a mixed loading_type that also clamps the shear strains to zero.",
                    UserWarning,
                    stacklevel=2,
                )
            return mechanical_bc

        quantities = terms(loading_type)
        if len(quantities) != 2:
            raise ValueError(f"{context}: invalid mixed loading_type {loading_type!r}.")
        if not isinstance(magnitude, list) or len(magnitude) != 2:
            raise TypeError(
                f"{context}: mixed loading_type {loading_type!r} requires a two-element magnitude list."
            )

        tensor_0 = read_tensor(magnitude[0], allow_x=True, context=f"{context}, magnitude[0] ({quantities[0]})")
        tensor_1 = read_tensor(magnitude[1], allow_x=True, context=f"{context}, magnitude[1] ({quantities[1]})")
        validate_mixed_complementarity(tensor_0, tensor_1, context)

        key_0 = damask_key(quantities[0])
        key_1 = damask_key(quantities[1])
        if key_0 == key_1:
            raise ValueError(f"{context}: mixed loading maps both quantities to {key_0!r}.")

        return {
            key_0: map_to_damask_tensor(tensor_0, quantities[0]),
            key_1: map_to_damask_tensor(tensor_1, quantities[1]),
        }

    # Select the first full-RVE FFT/spectral mechanical_BC entry.
    mechanical_BC = data.get("mechanical_BC", [])
    if not mechanical_BC:
        raise ValueError("No 'mechanical_BC' entry found in data.")
    if not isinstance(mechanical_BC, list):
        mechanical_BC = [mechanical_BC]

    bc_full_cube: _Optional[dict[str, _Any]] = None
    for bc in mechanical_BC:
        if not isinstance(bc, dict):
            continue
        vertex_list = bc.get("vertex_list", [])
        loading_type = bc.get("loading_type")
        if len(vertex_list) == 8 and loading_type in supported_loading_types:
            bc_full_cube = bc
            break

    if bc_full_cube is None:
        raise ValueError(
            "No FFT/spectral mechanical_BC entry found. Expected len(vertex_list) == 8 and "
            f"loading_type in {sorted(supported_loading_types)}."
        )

    loading_type = bc_full_cube.get("loading_type")
    loading_mode = bc_full_cube.get("loading_mode")
    if loading_mode not in supported_loading_modes:
        raise ValueError(
            f"Unsupported loading_mode={loading_mode!r}. Supported modes are {sorted(supported_loading_modes)}."
        )
    assert isinstance(loading_type, str)
    validate_units(loading_type)

    if loading_mode == "cyclic" and uses_rate(loading_type):
        raise ValueError(
            f"Cyclic loading is not supported for rate-controlled quantities. "
            f"loading_type={loading_type!r} contains a rate term (strain_rate/stress_rate), and the "
            "load ratio R = min/max is defined on absolute strain/stress extrema, not on rates. "
            "Use an absolute loading_type (strain, stress, or strain-stress) for cyclic loading, or "
            "use 'monotonic'/'static' mode for rate-controlled loading."
        )

    applied_load = bc_full_cube.get("applied_load", [])
    if not applied_load:
        raise ValueError("No 'applied_load' defined in selected mechanical_BC entry.")
    if not isinstance(applied_load, list):
        applied_load = [applied_load]

    loadsteps_grouped: list[list[dict[str, _Any]]] = []

    for i_load, load in enumerate(applied_load):
        if not isinstance(load, dict):
            raise TypeError(f"applied_load[{i_load}] must be a dictionary.")

        context = f"applied_load[{i_load}]"
        if "magnitude" not in load:
            raise ValueError(f"{context}: missing 'magnitude'.")
        if "duration" not in load:
            raise ValueError(f"{context}: missing 'duration'.")
        if "step" not in load:
            raise ValueError(f"{context}: missing 'step'.")

        magnitude = load["magnitude"]
        duration = float(load["duration"])
        step = float(load["step"])
        group: list[dict[str, _Any]] = []

        if loading_mode == "cyclic":
            # DAMASK has no cyclic keyword. A MiMeDat cyclic block is expanded into
            # ordinary DAMASK loadsteps following the closed, fully reversed convention
            #     0 -> max -> 0 -> min -> 0
            # i.e. four equal quarter-cycle loadsteps per cycle (target factors
            # 1 -> 0 -> R -> 0), repeated n_cycles times. Rate-controlled cyclic loading
            # is rejected earlier; only strain/stress/strain-stress reach this branch.
            if "frequency" not in load:
                raise ValueError(f"{context}: cyclic loading requires 'frequency'.")
            if "R" not in load:
                raise ValueError(f"{context}: cyclic loading requires load ratio 'R'.")

            frequency = float(load["frequency"])
            load_ratio = float(load["R"])
            if duration <= 0.0:
                raise ValueError(f"{context}: duration must be > 0, got {duration}.")
            if frequency <= 0.0:
                raise ValueError(f"{context}: frequency must be > 0, got {frequency}.")
            if step <= 0.0:
                raise ValueError(f"{context}: step must be > 0, got {step}.")
            if load_ratio >= 0.0:
                raise ValueError(
                    f"{context}: this cyclic convention (0 -> max -> 0 -> min -> 0) is defined for "
                    f"reversed loading and requires R = min/max < 0, got R={load_ratio}. "
                    "The 'magnitude' tensor is the peak (max); the valley is min = R * max."
                )

            # Number of full cycles that fit in 'duration'.
            n_cycles_float = duration * frequency
            n_cycles = int(round(n_cycles_float))
            if not _math.isclose(n_cycles_float, n_cycles, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise ValueError(
                    f"{context}: n_cycles = duration * frequency must be a positive integer. "
                    f"Got duration={duration}, frequency={frequency}, n_cycles={n_cycles_float}."
                )
            if n_cycles < 1:
                raise ValueError(
                    f"{context}: cyclic loading must contain at least one full cycle, "
                    f"got n_cycles={n_cycles}."
                )

            # Four equal quarter-cycle segments per cycle; N validated as segment_time/step.
            segment_time = 1.0 / (4.0 * frequency)
            N_segment = compute_N(segment_time, step, f"{context}, cyclic quarter-cycle")

            # Target factors for one cycle: 0 -> max -> 0 -> min -> 0.
            cycle_factors = [1.0, 0.0, load_ratio, 0.0]
            first_segment = True
            for i_cycle in range(n_cycles):
                for i_seg, factor in enumerate(cycle_factors):
                    seg_context = f"{context}, cycle {i_cycle}, segment {i_seg}"
                    target_magnitude = scale_magnitude_object(magnitude, factor, seg_context)
                    # Let per-tensor warnings (e.g. fully stress-controlled, mirrored
                    # shear) fire once, on the first built segment, instead of on every
                    # repeated segment.
                    if first_segment:
                        mechanical_bc = build_mechanical_bc(loading_type, target_magnitude, seg_context)
                        first_segment = False
                    else:
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            mechanical_bc = build_mechanical_bc(loading_type, target_magnitude, seg_context)
                    group.append({
                        "boundary_conditions": {"mechanical": mechanical_bc},
                        "discretization": {"t": segment_time, "N": N_segment},
                    })
        else:
            N = compute_N(duration, step, context)
            mechanical_bc = build_mechanical_bc(loading_type, magnitude, context)
            group.append({
                "boundary_conditions": {"mechanical": mechanical_bc},
                "discretization": {"t": duration, "N": N},
            })

        loadsteps_grouped.append(group)

    loadsteps = [loadstep for group in loadsteps_grouped for loadstep in group]

    def expand_control_list(values: _Optional[_Sequence[int]], name: str) -> _Optional[list[int]]:
        if values is None:
            return None
        values_list = [int(v) for v in values]
        n_generated = len(loadsteps)
        n_applied = len(applied_load)
        if len(values_list) == n_generated:
            return values_list
        if len(values_list) == n_applied:
            expanded: list[int] = []
            for value, group in zip(values_list, loadsteps_grouped):
                expanded.extend([value] * len(group))
            return expanded
        raise ValueError(
            f"Length mismatch for {name}: expected either {n_generated} generated DAMASK loadsteps "
            f"or {n_applied} applied_load entries, got {len(values_list)}."
        )

    f_out_values = expand_control_list(f_out_list, "f_out_list")
    f_restart_values = expand_control_list(f_restart_list, "f_restart_list")

    for i_step, loadstep in enumerate(loadsteps):
        if f_out_values is not None:
            loadstep["f_out"] = f_out_values[i_step]
        if f_restart_values is not None:
            loadstep["f_restart"] = f_restart_values[i_step]

    return loadsteps

def srepr(msg,
          glue: str = '\n',
          quote: bool = False) -> str:
    r"""
    Join (quoted) items with glue string.

    Parameters
    ----------
    msg : (sequence of) object with __repr__
        Items to join.
    glue : str, optional
        Glue used for joining operation. Defaults to '\n'.
    quote : bool, optional
        Quote items. Defaults to False.

    Returns
    -------
    joined : str
        String representation of the joined and quoted items.
    """
    q = '"' if quote else ''
    if (not hasattr(msg, 'strip') and
           (hasattr(msg, '__getitem__') or
            hasattr(msg, '__iter__'))):
        return glue.join(q+str(x)+q for x in msg)
    else:
        return q+(msg if isinstance(msg,str) else repr(msg))+q

def emph(msg) -> str:
    """
    Format with emphasis.

    Parameters
    ----------
    msg : (sequence of) object with __repr__
        Message to format.

    Returns
    -------
    formatted : str
        Formatted string representation of the joined items.
    """
    return _colors['bold']+srepr(msg)+_colors['end_color']

def deemph(msg) -> str:
    """
    Format with deemphasis.

    Parameters
    ----------
    msg : (sequence of) object with __repr__
        Message to format.

    Returns
    -------
    formatted : str
        Formatted string representation of the joined items.
    """
    return _colors['dim']+srepr(msg)+_colors['end_color']

def warn(msg) -> str:
    """
    Format for warning.

    Parameters
    ----------
    msg : (sequence of) object with __repr__
        Message to format.

    Returns
    -------
    formatted : str
        Formatted string representation of the joined items.
    """
    return _colors['warning']+emph(msg)+_colors['end_color']

def strikeout(msg) -> str:
    """
    Format as strikeout.

    Parameters
    ----------
    msg : (iterable of) object with __repr__
        Message to format.

    Returns
    -------
    formatted : str
        Formatted string representation of the joined items.
    """
    return _colors['crossout']+srepr(msg)+_colors['end_color']


def run(cmd: str,
        wd: str = './',
        env: _Optional[dict[str, str]] = None,
        timeout: _Optional[int] = None) -> stdioTuple:
    """
    Run a command.

    Parameters
    ----------
    cmd : str
        Command to be executed.
    wd : str, optional
        Working directory of process. Defaults to './'.
    env : dict, optional
        Environment for execution.
    timeout : int, optional
        Timeout in seconds.

    Returns
    -------
    stdout, stderr : (str, str)
        Output of the executed command.
    """
    def pass_signal(sig,_,proc,default):
        proc.send_signal(sig)
        _signal.signal(sig,default)
        _signal.raise_signal(sig)

    signals = [_signal.SIGINT,_signal.SIGTERM]

    logger.info(f"running '{cmd}' in '{wd}'")
    process = _subprocess.Popen(_shlex.split(cmd),
                                stdout = _subprocess.PIPE,
                                stderr = _subprocess.PIPE,
                                env = _os.environ if env is None else env,
                                cwd = wd,
                                encoding = 'utf-8')
    # ensure that process is terminated (https://stackoverflow.com/questions/22916783)
    sig_states = [_signal.signal(sig,_partial(pass_signal,proc=process,default=_signal.getsignal(sig))) for sig in signals]

    try:
        stdout,stderr = process.communicate(timeout=timeout)
    finally:
        for sig,state in zip(signals,sig_states):
            _signal.signal(sig,state)

    if process.returncode != 0:
        logger.error(stdout)
        logger.error(stderr)
        raise RuntimeError(f"'{cmd}' failed with returncode {process.returncode}")

    return stdioTuple(stdout, stderr)


@_contextlib.contextmanager
def open_text(fname: _FileHandle,
              mode: _Literal['r','w'] = 'r') -> _Generator[_TextIO, None, None]:                    # noqa
    """
    Open a text file with Unix line endings.

    If a path or string is given, a context manager ensures that
    the file handle is closed.
    If a file handle is given, it remains unmodified.

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        Name or handle of file.
    mode : {'r','w'}, optional
        Access mode: 'r'ead or 'w'rite, defaults to 'r'.

    Returns
    -------
    f : file handle
        File handle for a text file.
    """
    if isinstance(fname, (str,_Path)):
        fhandle = open(_Path(fname).expanduser(),mode,newline=('\n' if mode == 'w' else None))
        yield fhandle
        fhandle.close()
    else:
        yield fname


def time_stamp() -> str:
    """
    Provide current time as formatted string.

    Returns
    -------
    time_stamp : str
        Current time as string in %Y-%m-%d %H:%M:%S%z format.
    """
    return _datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z')

def execution_stamp(class_name: str,
                    function_name: _Optional[str] = None) -> str:
    """
    Timestamp the execution of a (function within a) class.

    execution_stamp : str
        Fingerprint of an operation: Class, (function), version, and
        current time.
    """
    _function_name = '' if function_name is None else f'.{function_name}'
    return f'damask.{class_name}{_function_name} v{_version} ({time_stamp()})'


def natural_sort(key: str) -> list[_Union[int, str]]:
    """
    Natural sort.

    For use in python's 'sorted'.

    References
    ----------
    https://en.wikipedia.org/wiki/Natural_sort_order
    """
    return [int(c) if c.isdigit() else c for c in _re.split('([0-9]+)', key)]


def show_progress(iterable: _Iterable,
                  N_iter: _Optional[int] = None,
                  prefix: str = '',
                  bar_length: int = 50) -> _Any:
    """
    Decorate a loop with a progress bar.

    Use similar like enumerate.

    Parameters
    ----------
    iterable : iterable
        Iterable to be decorated.
    N_iter : int, optional
        Total number of iterations. Required if iterable is not a sequence.
    prefix : str, optional
        Prefix string. Defaults to ''.
    bar_length : int, optional
        Length of progress bar in characters. Defaults to 50.
    """
    if isinstance(iterable,_abc.Sequence):
        if N_iter is None:
            N = len(iterable)
        else:
            raise ValueError('N_iter given for sequence')
    else:
        if N_iter is None:
            raise ValueError('N_iter not given')

        N = N_iter

    if N <= 1:
        for item in iterable:
            yield item
    else:
        status = ProgressBar(N,prefix,bar_length)
        for i,item in enumerate(iterable):
            yield item
            status.update(i)


def scale_to_coprime(v: _FloatSequence,
                     N_significant: int = 9) -> _np.ndarray:
    """
    Scale vector to co-prime (relatively prime) integers.

    Parameters
    ----------
    v : sequence of float, len (:)
        Vector to scale.
    N_significant : int, optional
        Number of significant digits to consider. Defaults to 9.

    Returns
    -------
    m : numpy.ndarray, shape (:)
        Vector scaled to co-prime numbers.
    """

    def get_square_denominator(x,max_denominator):
        """Denominator of the square of a number."""
        return _fractions.Fraction(x ** 2).limit_denominator(max_denominator).denominator

    def abs_lcm(a,b):
        """Absolute value of least common multiple."""
        return _np.abs(_np.lcm(a,b))

    max_denominator = int(10**(N_significant-1))

    v_ = _np.asarray(v)
    if _np.issubdtype(v_.dtype,_np.inexact):
        v_ = _np.round(_np.asarray(v,_np.float64)/_np.max(_np.abs(v)),N_significant)
    m = (v_ * _reduce(abs_lcm, map(lambda x: int(get_square_denominator(x,max_denominator)),v_))**0.5).astype(_np.int64)
    m = m//_reduce(_np.gcd,m)

    if not _np.allclose(m/_np.max(_np.abs(m)),v/_np.max(_np.abs(v)),atol=1e-2,rtol=0):
        raise ValueError(f'invalid result "{m}" for input "{v}"')

    return m


def project_equal_angle(vector: _np.ndarray,
                        direction: _Literal['x', 'y', 'z'] = 'z',                                   # noqa
                        normalize: bool = True,
                        keepdims: bool = False) -> _np.ndarray:
    """
    Apply equal-angle projection to vector.

    Parameters
    ----------
    vector : numpy.ndarray, shape (...,3)
        Vector coordinates to be projected.
    direction : {'x', 'y', 'z'}
        Projection direction. Defaults to 'z'.
    normalize : bool
        Ensure unit length of input vector. Defaults to True.
    keepdims : bool
        Maintain three-dimensional output coordinates.
        Defaults to False.

    Returns
    -------
    coordinates : numpy.ndarray, shape (...,2 | 3)
        Projected coordinates.

    Notes
    -----
    Two-dimensional output uses right-handed frame spanned by
    the next and next-next axis relative to the projection direction,
    e.g. x-y when projecting along z and z-x when projecting along y.

    Examples
    --------
    >>> import damask
    >>> import numpy as np
    >>> project_equal_angle(np.ones(3))
    array([0.3660, 0.3660])
    >>> project_equal_angle(np.ones(3),direction='x',normalize=False,keepdims=True)
    array([0. , 0.5, 0.5])
    >>> project_equal_angle([0,1,1],direction='y',normalize=True,keepdims=False)
    array([0.4142, 0. ])
    """
    shift = 'zyx'.index(direction)
    v = _np.roll(vector/_np.linalg.norm(vector,axis=-1,keepdims=True) if normalize else vector,
                 shift,axis=-1)
    return _np.roll(_np.block([v[...,:2]/(1.0+_np.abs(v[...,2:3])),_np.zeros_like(v[...,2:3])]),
                    -shift if keepdims else 0,axis=-1)[...,:3 if keepdims else 2]

def project_equal_area(vector: _np.ndarray,
                       direction: _Literal['x', 'y', 'z'] = 'z',                                    # noqa
                       normalize: bool = True,
                       keepdims: bool = False) -> _np.ndarray:
    """
    Apply equal-area projection to vector.

    Parameters
    ----------
    vector : numpy.ndarray, shape (...,3)
        Vector coordinates to be projected.
    direction : {'x', 'y', 'z'}
        Projection direction. Defaults to 'z'.
    normalize : bool
        Ensure unit length of input vector. Defaults to True.
    keepdims : bool
        Maintain three-dimensional output coordinates.
        Defaults to False.

    Returns
    -------
    coordinates : numpy.ndarray, shape (...,2 | 3)
        Projected coordinates.

    Notes
    -----
    Two-dimensional output uses right-handed frame spanned by
    the next and next-next axis relative to the projection direction,
    e.g. x-y when projecting along z and z-x when projecting along y.

    Examples
    --------
    >>> import damask
    >>> import numpy as np
    >>> project_equal_area(np.ones(3))
    array([0.4597, 0.4597])
    >>> project_equal_area(np.ones(3),direction='x',normalize=False,keepdims=True)
    array([0. , 0.7071, 0.7071])
    >>> project_equal_area([0,1,1],direction='y',normalize=True,keepdims=False)
    array([0.5412, 0. ])
    """
    shift = 'zyx'.index(direction)
    v = _np.roll(vector/_np.linalg.norm(vector,axis=-1,keepdims=True) if normalize else vector,
                 shift,axis=-1)
    return _np.roll(_np.block([v[...,:2]/_np.sqrt(1.0+_np.abs(v[...,2:3])),_np.zeros_like(v[...,2:3])]),
                    -shift if keepdims else 0,axis=-1)[...,:3 if keepdims else 2]


def hybrid_IA(dist: _FloatSequence,
              N: int,
              rng_seed: _Optional[_NumpyRngSeed] = None) -> _np.ndarray:
    """
    Hybrid integer approximation.

    Parameters
    ----------
    dist : numpy.ndarray
        Distribution to be approximated.
    N : int
        Number of samples to draw.
    rng_seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the BitGenerator. Defaults to None.
        If None, then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    hist : numpy.ndarray, shape (N)
        Integer approximation of the distribution.
    """
    N_opt_samples = _np.maximum(_np.count_nonzero(dist),N)                                          # random subsampling if too little samples requested
    N_inv_samples = _np.int_(0)

    scale_,scale,inc_factor = (0.0,float(N_opt_samples),1.0)
    while (not _np.isclose(scale, scale_)) and (N_inv_samples != N_opt_samples):
        repeats = _np.rint(scale*_np.array(dist)).astype(_np.int64)
        N_inv_samples = _np.sum(repeats)
        scale_,scale,inc_factor = (scale,scale+inc_factor*0.5*(scale - scale_), inc_factor*2.0) \
                                   if N_inv_samples < N_opt_samples else \
                                  (scale_,0.5*(scale_ + scale), 1.0)

    return _np.repeat(_np.arange(len(dist)),repeats)[_np.random.default_rng(rng_seed).permutation(N_inv_samples)[:N]]


def shapeshifter(fro: tuple[int, ...],
                 to: tuple[int, ...],
                 mode: _Literal['left','right'] = 'left',                                           # noqa
                 keep_ones: bool = False) -> tuple[int, ...]:
    """
    Return dimensions that reshape 'fro' to become broadcastable to 'to'.

    Parameters
    ----------
    fro : tuple
        Original shape of array.
    to : tuple
        Target shape of array after broadcasting.
        len(to) cannot be less than len(fro).
    mode : {'left', 'right'}, optional
        Indicates whether new axes are preferably added to
        either left or right of the original shape.
        Defaults to 'left'.
    keep_ones : bool, optional
        Treat '1' in fro as literal value instead of dimensional placeholder.
        Defaults to False.

    Returns
    -------
    new_dims : tuple
        Dimensions for reshape.

    Examples
    --------
    >>> import numpy as np
    >>> from damask import util
    >>> a = np.ones((3,4,2))
    >>> b = np.ones(4)
    >>> b_extended = b.reshape(util.shapeshifter(b.shape,a.shape))
    >>> (a * np.broadcast_to(b_extended,a.shape)).shape
    (3, 4, 2)
    """
    if len(fro) == 0 and len(to) == 0: return tuple()
    _fro = [1] if len(fro) == 0 else list(fro)[::-1 if mode=='left' else 1]
    _to  = [1] if len(to)  == 0 else list(to) [::-1 if mode=='left' else 1]

    final_shape: list[int] = []
    index = 0
    for i,item in enumerate(_to):
        if item == _fro[index]:
            final_shape.append(item)
            index+=1
        else:
            final_shape.append(1)
            if _fro[index] == 1 and not keep_ones:
                index+=1
        if index == len(_fro):
            final_shape = final_shape+[1]*(len(_to)-i-1)
            break
    if index != len(_fro): raise ValueError(f'shapes cannot be shifted {fro} --> {to}')
    return tuple(final_shape[::-1] if mode == 'left' else final_shape)

def shapeblender(a: tuple[int, ...],
                 b: tuple[int, ...],
                 keep_ones: bool = False) -> tuple[int, ...]:
    """
    Calculate shape that overlaps the rightmost entries of 'a' with the leftmost of 'b'.

    Parameters
    ----------
    a : tuple
        Shape of first ("left") array.
    b : tuple
        Shape of second ("right") array.
    keep_ones : bool, optional
        Treat innermost '1's as literal value instead of dimensional placeholder.
        Defaults to False.

    Returns
    -------
    shape : tuple
        Shape that overlaps the rightmost entries of 'a' with the leftmost of 'b'.

    Examples
    --------
    >>> shapeblender((3,2),(3,2))
    (3, 2)
    >>> shapeblender((4,3),(3,2))
    (4, 3, 2)
    >>> shapeblender((4,4),(3,2))
    (4, 4, 3, 2)
    >>> shapeblender((1,2),(1,2,3))
    (1, 2, 3)
    >>> shapeblender((),(2,2,1))
    (2, 2, 1)
    >>> shapeblender((1,),(2,2,1))
    (2, 2, 1)
    >>> shapeblender((1,),(2,2,1),True)
    (1, 2, 2, 1)
    """
    def is_broadcastable(a,b):
        try:
            _np.broadcast_shapes(a,b)
            return True
        except ValueError:
            return False

    a_,_b = a,b
    if keep_ones:
        i = min(len(a_),len(_b))
        while i > 0 and a_[-i:] != _b[:i]: i -= 1
        return a_ + _b[i:]
    else:
        a_ += max(0,len(_b)-len(a_))*(1,)
        while not is_broadcastable(a_,_b):
            a_ = a_ + ((1,) if len(a_)<=len(_b) else ())
            _b = ((1,) if len(_b)<len(a_) else ()) + _b
        return _np.broadcast_shapes(a_,_b)


def DREAM3D_base_group(fname: _Union[str, _Path, _h5py.File]) -> str:
    """
    Determine the base group of a DREAM.3D file.

    The base group is defined as the group (folder) that contains
    a 'SPACING' dataset in a '_SIMPL_GEOMETRY' group.

    Parameters
    ----------
    fname : str, pathlib.Path, or _h5py.File
        Filename of the DREAM.3D (HDF5) file.

    Returns
    -------
    path : str
        Path to the base group.
    """
    def get_base_group(f: _h5py.File) -> str:
        base_group = f.visit(lambda path: path.rsplit('/',2)[0] if '_SIMPL_GEOMETRY/SPACING' in path else None)
        if base_group is None:
            raise ValueError(f'could not determine base group in file "{fname}"')
        return base_group

    if isinstance(fname,_h5py.File):
        return get_base_group(fname)

    with _h5py.File(_Path(fname).expanduser(),'r') as f:
        return get_base_group(f)

def DREAM3D_cell_data_group(fname: _Union[str, _Path, _h5py.File]) -> str:
    """
    Determine the cell data group of a DREAM.3D file.

    The cell data group is defined as the group (folder) that contains
    a dataset in the base group whose length matches the total number
    of points as specified in '_SIMPL_GEOMETRY/DIMENSIONS'.

    Parameters
    ----------
    fname : str, pathlib.Path, or h5py.File
        Filename of the DREAM.3D (HDF5) file.

    Returns
    -------
    path : str
        Path to the cell data group.
    """
    def get_cell_data_group(f: _h5py.File) -> str:
        base_group = DREAM3D_base_group(f)
        cells = tuple(f['/'.join([base_group,'_SIMPL_GEOMETRY','DIMENSIONS'])][()][::-1])
        cell_data_group = f[base_group].visititems(lambda path,obj: path.split('/')[0] \
                                                   if isinstance(obj,_h5py._hl.dataset.Dataset) and _np.shape(obj)[:-1] == cells \
                                                   else None)
        if cell_data_group is None:
            raise ValueError(f'could not determine cell-data group in file "{fname}/{base_group}"')
        return cell_data_group

    if isinstance(fname,_h5py.File):
        return get_cell_data_group(fname)

    with _h5py.File(_Path(fname).expanduser(),'r') as f:
        return get_cell_data_group(f)


def _standardize_MillerBravais(idx: _IntSequence) -> _np.ndarray:
    """
    Convert Miller-Bravais indices with missing component to standard (full) form.

    Parameters
    ----------
    idx : numpy.ndarray, shape (...,4) or (...,3)
        Miller–Bravais indices of crystallographic direction [uvtw] or plane normal (hkil).
        The third index (t or i) can be omitted completely or given as "..." (Ellipsis).

    Returns
    -------
    uvtw|hkil : numpy.ndarray, shape (...,4)
        Miller-Bravais indices of [uvtw] direction or (hkil) plane normal.
    """
    def expand(v: _np.ndarray) -> _np.ndarray:
        """Expand from 3 to 4 indices."""
        return _np.block([v[...,:2], -_np.sum(v[...,:2],axis=-1,keepdims=True), v[...,2:]])

    a = _np.asarray(idx)
    if _np.issubdtype(a.dtype,_np.signedinteger):
        if a.shape[-1] == 4:
            if (_np.sum(a[...,:3],axis=-1) != 0).any(): raise ValueError(rf'u+v+t≠0 | h+k+i≠0: {a}')
            return a
        elif a.shape[-1] == 3:
            return expand(a)
    else:
        if a.shape[-1] == 4:
            b = (_np.block([a[...,:2],
                            _np.where(a[...,2:3] == ..., -_np.sum(a[...,:2],axis=-1,keepdims=True),a[...,2:3]),
                            a[...,3:]]))
            if (_np.sum(b[...,:3].astype(int),axis=-1) != 0).any(): raise ValueError(rf'u+v+t≠0 | h+k+i≠0: {b}')
        elif a.shape[-1] == 3:
            b = expand(a)

        if (b != (c := b.astype(int))).any():
            raise ValueError(f'"uvtw" | "hkil" are not (castable to) signed integers: {a}')
        return c

    raise ValueError(f'invalid Miller-Bravais indices {a}')


def Bravais_to_Miller(*,
                      uvtw: _Optional[_IntSequence] = None,
                      hkil: _Optional[_IntSequence] = None) -> _np.ndarray:                         # numpydoc ignore=PR01,PR02
    """
    Transform 4 Miller–Bravais indices to 3 Miller indices of crystal direction [uvw] or plane normal (hkl).

    Parameters
    ----------
    uvtw|hkil : numpy.ndarray, shape (...,4) or (...,3)
        Miller–Bravais indices of crystallographic direction [uvtw] or plane normal (hkil).
        The third index (t or i) can be omitted completely or given as "..." (Ellipsis).

    Returns
    -------
    uvw|hkl : numpy.ndarray, shape (...,3)
        Miller indices of [uvw] direction or (hkl) plane normal.
    """
    if (uvtw is not None) ^ (hkil is None):
        raise KeyError('specify either "uvtw" or "hkil"')
    elif uvtw is not None:
        axis,basis = _standardize_MillerBravais(uvtw),_np.array([[2,1,0,0],
                                                                 [1,2,0,0],
                                                                 [0,0,0,1]])
    elif hkil is not None:
        axis,basis = _standardize_MillerBravais(hkil),_np.array([[1,0,0,0],
                                                                 [0,1,0,0],
                                                                 [0,0,0,1]])
    uvw_hkl = _np.einsum('il,...l',basis,axis)

    return uvw_hkl//_np.gcd.reduce(uvw_hkl,axis=-1,keepdims=True)


MillerBravais_to_Miller = Bravais_to_Miller


def Miller_to_Bravais(*,
                      uvw: _Optional[_IntSequence] = None,
                      hkl: _Optional[_IntSequence] = None) -> _np.ndarray:                          # numpydoc ignore=PR01,PR02
    """
    Transform 3 Miller indices to 4 Miller–Bravais indices of crystal direction [uvtw] or plane normal (hkil).

    Parameters
    ----------
    uvw|hkl : numpy.ndarray, shape (...,3)
        Miller indices of crystallographic direction [uvw] or plane normal (hkl).

    Returns
    -------
    uvtw|hkil : numpy.ndarray, shape (...,4)
        Miller–Bravais indices of [uvtw] direction or (hkil) plane normal.
    """
    if (uvw is not None) ^ (hkl is None):
        raise KeyError('specify either "uvw" or "hkl"')
    axis,basis = (_np.asarray(uvw),_np.array([[ 2,-1, 0],
                                              [-1, 2, 0],
                                              [-1,-1, 0],
                                              [ 0, 0, 3]])) \
                 if hkl is None else \
                 (_np.asarray(hkl),_np.array([[ 1, 0, 0],
                                              [ 0, 1, 0],
                                              [-1,-1, 0],
                                              [ 0, 0, 1]]))
    if (axis != axis.astype(int)).any():
        raise ValueError(f'"uvt" | "hki" are not (castable to) signed integers: {axis}')
    uvtw_hkil = _np.einsum('il,...l',basis,axis.astype(int))

    return uvtw_hkil//_np.gcd.reduce(uvtw_hkil,axis=-1,keepdims=True)


Miller_to_MillerBravais = Miller_to_Bravais


def dict_prune(d: dict) -> dict:
    """
    Recursively remove empty dictionaries.

    Parameters
    ----------
    d : dict
        Dictionary to prune.

    Returns
    -------
    pruned : dict
        Pruned dictionary.
    """
    # https://stackoverflow.com/questions/48151953
    new = {}
    for k,v in d.items():
        if isinstance(v, dict):
            v = dict_prune(v)
        if not isinstance(v,dict) or v != {}:
            new[k] = v

    return new

def dict_flatten(d: dict) -> dict:
    """
    Recursively remove keys of single-entry dictionaries.

    Parameters
    ----------
    d : dict
        Dictionary to flatten.

    Returns
    -------
    flattened : dict
        Flattened dictionary.
    """
    if isinstance(d,dict) and len(d) == 1:
        entry = d[list(d.keys())[0]]
        new = dict_flatten(entry.copy()) if isinstance(entry,dict) else entry
    else:
        new = {k: (dict_flatten(v) if isinstance(v, dict) else v) for k,v in d.items()}

    return new


def to_list(a: _Any) -> list:
    """
    Put into list.

    Parameters
    ----------
    a : any
        Variable to put into list or convert to list.

    Returns
    -------
    l : list
        Data in list.
    """
    return [a] if not hasattr(a,'__iter__') or isinstance(a,str) else list(a)


####################################################################################################
# Classes
####################################################################################################
class ProgressBar:
    """
    Report progress of an interation as a status bar.

    Works for 0-based loops, ETA is estimated by linear extrapolation.
    """

    def __init__(self,
                 total: int,
                 prefix: str,
                 bar_length: int):
        """
        New progress bar.

        Parameters
        ----------
        total : int
            Total # of iterations.
        prefix : str
            Prefix string.
        bar_length : int
            Character length of bar.
        """
        self.total = total
        self.prefix = prefix
        self.bar_length = bar_length
        self.time_start = self.time_last_update = _datetime.datetime.now()
        self.fraction_last = 0.0

        if _sys.stdout.isatty():
            _sys.stdout.write(f"{self.prefix} {'░'*self.bar_length}   0% ETA n/a")

    def update(self,
               iteration: int) -> None:

        fraction = (iteration+1) / self.total

        if (filled_length := int(self.bar_length * fraction)) > int(self.bar_length * self.fraction_last) or \
            _datetime.datetime.now() - self.time_last_update > _datetime.timedelta(seconds=10):
            self.time_last_update = _datetime.datetime.now()
            bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
            remaining_time = (_datetime.datetime.now() - self.time_start) \
                           * (self.total - (iteration+1)) / (iteration+1)
            remaining_time -= _datetime.timedelta(microseconds=remaining_time.microseconds)         # remove μs
            if _sys.stdout.isatty():
                _sys.stdout.write(f'\r{self.prefix} {bar} {fraction:>4.0%} ETA {remaining_time}')

        self.fraction_last = fraction

        if iteration == self.total - 1 and _sys.stdout.isatty():
            _sys.stdout.write('\n')
