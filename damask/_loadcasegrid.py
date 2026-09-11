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
        f_out_list: Optional[Sequence[int]] = None,
        f_restart_list: Optional[Sequence[int]] = None,
        mechanical_solver: str = "spectral_basic",
    ) -> "LoadcaseGrid":
        """
        Construct a DAMASK LoadcaseGrid from a MiMeDat mechanical_BC block.

        Accepts a parsed mapping, a JSON file path, or a raw JSON string. The actual
        MiMeDat-to-DAMASK conversion is delegated to
        ``_build_loadsteps_from_mechanical_bc``.
        """
        if isinstance(src, Mapping):
            data = dict(src)
        elif isinstance(src, Path):
            if not src.exists() or not src.is_file():
                raise ValueError(f"No JSON file found at path: {src}")
            data = json.loads(src.read_text())
        elif isinstance(src, str):
            src_str = src.strip()
            if src_str.startswith("{") or src_str.startswith("["):
                try:
                    data = json.loads(src_str)
                except json.JSONDecodeError as exc:
                    raise ValueError("Input string could not be parsed as JSON.") from exc
            else:
                path = Path(src_str)
                if not path.exists() or not path.is_file():
                    raise ValueError("Input string is neither raw JSON nor a valid JSON file path.")
                data = json.loads(path.read_text())
        else:
            raise TypeError("src must be a file path, raw JSON string, or dict-like Mapping.")

        loadsteps = _build_loadsteps_from_mechanical_bc(
            data=data,
            f_out_list=f_out_list,
            f_restart_list=f_restart_list,
        )

        return cls(
            solver={"mechanical": mechanical_solver},
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
