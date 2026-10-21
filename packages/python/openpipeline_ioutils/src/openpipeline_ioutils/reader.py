from __future__ import annotations

import importlib
import zipfile
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
  import anndata


def _is_h5(source: Path) -> bool:
    signature = b"\x89HDF\r\n\x1a\n"
    offset = 0
    with source.open("rb") as f:
        chunk = f.read(len(signature))
        while chunk:
            if chunk == signature:
                return True
            offset = 512 if offset == 0 else offset * 2
            f.seek(offset)
            chunk = f.read(len(signature))
    return False        

def _detect_format_h5(source: str | Path) -> Literal["MuData", "AnnData"]:
    import h5py
    with h5py.File(source, "r") as open_source:
        if open_source.attrs.get("encoding-type") == "anndata":
            return "AnnData"
        if open_source.attrs.get("encoding-type") == "MuData":
            return "MuData"
        raise RuntimeError(f"Could not detect format for H5 file '{source}'. Possible formats are 'MuData' or 'AnnData'.")

def _detect_format_zarr(source: str | Path) -> Literal["MuData", "AnnData", "SpatialData"]:
    import zarr
    open_zarr = zarr.open_group(source, zarr_format=3, use_consolidated=False, mode="r")
    group_attributes = open_zarr.metadata.attributes
    if "spatialdata_attrs" in group_attributes:
        return "SpatialData"
    if group_attributes.get("encoding-type") == "anndata":
        return "AnnData"
    if group_attributes.get("encoding-type") == "MuData":
        return "MuData"
    raise RuntimeError(f"Could not detect format for zarr file '{source}'. Possible format are 'MuData' or 'AnnData' or 'SpatialData'.")


def _detect_format(source: str | Path, storage_backend: Literal["h5", "zarr"]) -> Literal["AnnData", "MuData", "SpatialData"]:
    HANDLERS = {
        "h5": _detect_format_h5,
        "zarr": _detect_format_zarr
    }
    try:
        return HANDLERS[storage_backend](source)
    except KeyError:
        raise ValueError(f"Backend {storage_backend} is not supported.") from None
        

def _detect_storage_backend(source: str | Path) -> Literal["zarr", "h5"]:
    source_path = Path(source)
    # We only support local files or directories, even though zarr allows remote storage.
    if not source_path.exists():
        raise FileNotFoundError(f"{source}: no such file or directory.")
    if source_path.is_dir():
        if (source_path / ".zgroup").is_file():
            raise RuntimeError(
                f"'{source}' is a zarr v2 store. Only zarr v3 is supported; "
                "rewrite the data with zarr_format=3 (the default since anndata 0.13)."
            )
        if (source_path / "zarr.json").is_file():
            return "zarr"
        raise RuntimeError("A directory was provided, but it does not seem to provide zarr data.")
    if zipfile.is_zipfile(source):
        raise NotImplementedError(
            f"'{source}' is a zip file. Zipped zarr stores are not supported; "
            "extract the archive and pass the directory instead."
        )
    if _is_h5(source_path):
        return "h5"
    raise NotImplementedError("Could not determine storage backend. If you wish to use a remote .zarr store, it is not supported.")

def _import_optional(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        e.add_note(f"Reading this file requires '{module_name}'.")
        raise


def _read_spatialdata(source: str | Path, key: str | None, **kwargs: Any) -> anndata.AnnData:
    spatialdata = _import_optional("spatialdata")
    if key is None:
        raise ValueError("When reading a SpatialData object, `key` must be set to the name of a table.")
    tables = spatialdata.read_zarr(source, **kwargs).tables
    if key not in tables:
        raise KeyError(f"Table '{key}' was not found in '{source}'. Available tables: {list(tables)}.")
    return tables[key]


def _read_mudata_h5(source: str, key: str, **kwargs: Any) -> anndata.AnnData:
    mudata = _import_optional("mudata")
    try:
        return mudata.read_h5ad(source, mod=key, **kwargs)
    except KeyError as e:
        raise KeyError(f"Modality '{key}' was not found in '{source}'.") from e


def _read_mudata_zarr(source: str, key: str, **kwargs: Any) -> anndata.AnnData:
    import anndata
    import zarr
    root = zarr.open_group(source, mode="r")
    modalities = root["mod"]
    if not isinstance(modalities, zarr.Group):
        raise TypeError(f"Expected 'mod' to be a zarr group in '{source}'.")
    if key not in modalities:
        raise KeyError(f"Modality '{key}' was not found in '{source}'.")
    modality = modalities[key]
    if not isinstance(modality, zarr.Group):
        raise TypeError(f"Expected modality '{key}' to be a zarr group in '{source}'.")
    return anndata.read_zarr(modality, **kwargs)


def _read_mudata(source: str | Path, key: str | None, storage_backend: Literal["zarr", "h5"], **kwargs: Any) -> anndata.AnnData:
    if key is None:
        raise ValueError("When reading a MuData object, `key` must be set to the name of a modality.")
    source_str = str(source) # TODO: remove when mudata fixes Path bug
    FORMAT_DISPATCHER = {
        "h5": _read_mudata_h5, 
        "zarr": _read_mudata_zarr,
    }
    return FORMAT_DISPATCHER[storage_backend](source_str, key, **kwargs)


def _read_anndata(source: str | Path, key: str | None, storage_backend: Literal["zarr", "h5"], **kwargs: Any) -> anndata.AnnData:
    import anndata
    if key is not None:
        raise ValueError("When reading an AnnData object, `key` should not be set.")
    if storage_backend == "zarr":
        return anndata.read_zarr(source, **kwargs)
    return anndata.read_h5ad(source, **kwargs)


def _read(source: str | Path,
          data_format: Literal["MuData", "AnnData", "SpatialData"], 
          storage_backend: Literal["zarr", "h5"],
          key: str | None = None,
          **kwargs: Any) -> anndata.AnnData:
    
    READERS = {
        "MuData": partial(_read_mudata, key = key, storage_backend=storage_backend),
        "AnnData": partial(_read_anndata, key = key, storage_backend=storage_backend),
        "SpatialData": partial(_read_spatialdata, key = key) # only zarr...
    }
    return READERS[data_format](source, **kwargs)


def read_annotated_array(source: str | Path, key: str | None = None, **kwargs: Any) -> anndata.AnnData:
    """
    Read an AnnData object from a variety of storage backends and data formats.
    Dependencies are not loaded until the time of reading the data so that only the 
    library for the specific format is imported.
    """
    storage_backend = _detect_storage_backend(source)
    data_format = _detect_format(source, storage_backend)
    return _read(source, data_format=data_format, storage_backend=storage_backend, key=key, **kwargs)
