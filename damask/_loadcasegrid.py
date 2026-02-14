from typing import Optional, Union, Any, Mapping, Sequence

from pathlib import Path
import json
from numpy import ma

from ._typehints import FileHandle
from ._yaml import NiceDumper
from . import YAML
from .util import _build_loadsteps_from_mechanical_bc


class MaskedMatrixDumper(NiceDumper):
    """Format masked matrices."""

    def represent_data(self, data: Any):
        return super().represent_data(data.astype(object).filled('x')                               # type: ignore[attr-defined]
                                      if isinstance(data, ma.core.MaskedArray) else
                                      data)


class LoadcaseGrid(YAML):
    """Load case for grid solver."""

    def __init__(self,
                 config: Optional[Union[str,dict[str,Any]]] = None,
                 *,
                 solver: Optional[dict[str,str]] = None,
                 loadstep: Optional[list[dict[str,Any]]] = None):
        """
        New grid solver load case.

        Parameters
        ----------
        config : dict or str, optional
            Grid solver load case. String needs to be valid YAML.
        solver : dict, optional
            Solver configuration.
            Defaults to an empty dict if 'config' is not given.
        loadstep : list of dict, optional
            Load step configuration.
            Defaults to an empty list if 'config' is not given.
        """
        kwargs: dict[str,Union[dict[str,str],list[dict[str,Any]]]] = {}
        default: Union[list,dict]
        for arg,value,default in [('solver',solver,{}),('loadstep',loadstep,[])]:                   # type: ignore[assignment]
            if value is not None:
                kwargs[arg] = value
            elif config is None:
                kwargs[arg] = default

        super().__init__(config,**kwargs)

    @classmethod
    def from_mechanical_bc(
            cls,
            src: Union[str, Path, Mapping[str, Any]],
            N_list: Sequence[int],
            f_out_list: Sequence[int],
            f_restart_list: Sequence[int],
            mechanical_solver: str = "spectral_basic", ) -> "LoadcaseGrid":
        """
        Construct a LoadcaseGrid from a mechanical boundary condition JSON.

        The generated loadcase has the form::

            solver: {mechanical: <mechanical_solver>}
            loadstep:
            - boundary_conditions:
                mechanical:
                  dot_F: [[...], [...], [...]]
                  P:     [[...], [...], [...]]
              discretization: {t: <duration>, N: <N>}
              f_out: <int>
              f_restart: <int>

        Parameters
        ----------
        src : str, pathlib.Path, or mapping
            Source of the boundary-condition data. One of:
              - Path to a JSON file.
              - Raw JSON string.
              - A dict-like object containing the parsed JSON.
            The data is expected to contain a key ``"mechanical_BC"`` as
            interpreted by ``_build_loadsteps_from_mechanical_bc``.
        N_list : sequence of int
            Number of increments per load step.
        f_out_list : sequence of int
            Output frequency for each load step.
        f_restart_list : sequence of int
            Restart frequency for each load step.
        mechanical_solver : str, optional
            Name of the mechanical solver. This is stored as::

                solver: {mechanical: <mechanical_solver>}

            Default is ``"spectral_basic"``.

        Returns
        -------
        LoadcaseGrid
            A LoadcaseGrid instance containing the solver configuration
            and the constructed loadsteps.
        """
        # --- accept path, JSON string, or dict ---
        if isinstance(src, (str, Path)):
            src_str = str(src)
            try:
                p = Path(src_str)
                if p.exists() and p.is_file():
                    data = json.loads(p.read_text())
                else:
                    data = json.loads(src_str)  # treat as raw JSON string
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Input string could not be parsed as JSON and no file was found at the path."
                ) from e
        elif isinstance(src, Mapping):
            data = dict(src)  # shallow copy
        else:
            raise TypeError(
                "src must be a file path (str/Path), a JSON string, or a dict-like Mapping."
            )

        # Reuse the helper that builds the list[dict] loadsteps
        loadsteps = _build_loadsteps_from_mechanical_bc(
            data=data,
            N_list=N_list,
            f_out_list=f_out_list,
            f_restart_list=f_restart_list,
        )

        # Match the desired YAML structure
        solver_cfg: dict[str, str] = {"mechanical": mechanical_solver}

        return cls(
            solver=solver_cfg,
            loadstep=loadsteps,
        )

    def save(self,
             fname: FileHandle,
             **kwargs):
        """
        Save to YAML file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            Filename or file to write.
        **kwargs : dict
            Keyword arguments parsed to yaml.dump.
        """
        if 'Dumper' not in kwargs:
            kwargs['Dumper'] = MaskedMatrixDumper

        super().save(fname=fname,**kwargs)
