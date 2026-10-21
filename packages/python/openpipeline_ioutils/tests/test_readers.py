import importlib
import shutil
import sys
from uuid import uuid4

import anndata as ad
import geopandas as gpd
import mudata as md
import pandas as pd
import pytest
import zarr
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from openpipeline_ioutils.reader import read_annotated_array

ad.settings.zarr_write_format = 3

@pytest.fixture(params=["h5", "zarr"])
def storage_backend(request):
    return request.param


@pytest.fixture
def small_anndata_1():
    df = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], index=["obs1", "obs2"], columns=["var1", "var2", "var3"]
    )
    obs = pd.DataFrame([["A"], ["B"]], index=df.index, columns=["Obs"])
    var = pd.DataFrame([["a"], ["b"], ["c"]], index=df.columns, columns=["Feat"])
    ad1 = ad.AnnData(df, obs=obs, var=var)
    return ad1


@pytest.fixture
def small_anndata_2():
    df = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], index=["obs1", "obs2"], columns=["var4", "var5", "var6"]
    )
    obs2 = pd.DataFrame(["C", "D"], index=df.index, columns=["Obs"])
    var2 = pd.DataFrame(["d", "e", "g"], index=df.columns, columns=["Feat"])
    ad2 = ad.AnnData(df, obs=obs2, var=var2)
    return ad2


@pytest.fixture
def input_h5mu(small_anndata_1, small_anndata_2):
    return md.MuData({"mod1": small_anndata_1, "mod2": small_anndata_2})


@pytest.fixture
def small_spatialdata(small_anndata_1):
    circles = gpd.GeoDataFrame(
        {"radius": [1.0, 1.0]},
        geometry=[Point(0, 0), Point(10, 10)],
        index=small_anndata_1.obs_names,
    )
    table = small_anndata_1.copy()
    table.obs["region"] = pd.Categorical(["circles"] * table.n_obs)
    table.obs["instance_id"] = table.obs_names
    table = TableModel.parse(
        table, region="circles", region_key="region", instance_key="instance_id"
    )
    return SpatialData(
        shapes={"circles": ShapesModel.parse(circles)}, tables={"table1": table}
    )


@pytest.fixture
def random_path(tmp_path):
    def wrapper(extension=None):
        extension = "" if not extension else f".{extension}"
        return tmp_path / f"{uuid4()}{extension}"

    return wrapper

@pytest.fixture
def small_anndata_1_path(small_anndata_1, random_path, storage_backend):
    random_anndata = random_path("h5ad" if storage_backend == "h5" else "zarr")
    writer = small_anndata_1.write_h5ad if storage_backend == "h5" else small_anndata_1.write_zarr
    writer(random_anndata)
    return random_anndata

@pytest.fixture
def small_mudata_path(input_h5mu, random_path, storage_backend):    
    random_mudata = random_path("h5mu" if storage_backend == "h5" else "zarr")
    writer = input_h5mu.write_h5mu if storage_backend == "h5" else input_h5mu.write_zarr
    writer(random_mudata)
    return random_mudata

@pytest.fixture
def small_spatialdata_path(small_spatialdata, random_path):
    # spatialdata does not support the h5 backend
    random_spatialdata = random_path("zarr")
    small_spatialdata.write(random_spatialdata)
    return random_spatialdata

def test_read_annotated_array_is_exported():
    import openpipeline_ioutils
    assert openpipeline_ioutils.read_annotated_array is read_annotated_array

def test_read_anndata(small_anndata_1_path):
    output = read_annotated_array(small_anndata_1_path)
    assert isinstance(output, ad.AnnData)

def test_read_mudata(small_mudata_path):
    output = read_annotated_array(small_mudata_path, key="mod1")
    assert isinstance(output, ad.AnnData)

def test_read_spatialdata(small_spatialdata_path):
    output = read_annotated_array(small_spatialdata_path, key="table1")
    assert isinstance(output, ad.AnnData)

@pytest.fixture
def zipped(tmp_path):
    def wrapper(zarr_path):
        # Zip the *contents* of the store so that zarr.json sits at the archive root.
        return shutil.make_archive(str(tmp_path / f"{uuid4()}"), "zip", root_dir=zarr_path)

    return wrapper

@pytest.mark.parametrize("storage_backend", ["zarr"])
def test_read_zip_raises(small_anndata_1_path, zipped):
    with pytest.raises(NotImplementedError, match="Zipped zarr stores are not supported"):
        read_annotated_array(zipped(small_anndata_1_path))

def test_read_zarr_v2_raises(tmp_path):
    zarr.open_group(tmp_path, mode="w", zarr_format=2)
    with pytest.raises(RuntimeError, match="Only zarr v3 is supported"):
        read_annotated_array(tmp_path)

def test_read_non_zarr_directory_raises(tmp_path):
    with pytest.raises(RuntimeError, match="does not seem to provide zarr data"):
        read_annotated_array(tmp_path)

def test_read_anndata_with_key_raises(small_anndata_1_path):
    with pytest.raises(ValueError, match="should not be set"):
        read_annotated_array(small_anndata_1_path, key="error")

def test_read_mudata_without_key_raises(small_mudata_path):
    with pytest.raises(ValueError, match="must be set to the name of a modality"):
        read_annotated_array(small_mudata_path)

def test_read_mudata_nonexistent_key_raises(small_mudata_path):
    with pytest.raises(KeyError, match="Modality 'error' was not found"):
        read_annotated_array(small_mudata_path, key="error")

def test_read_spatialdata_without_key_raises(small_spatialdata_path):
    with pytest.raises(ValueError, match="must be set to the name of a table"):
        read_annotated_array(small_spatialdata_path)

def test_read_spatialdata_nonexistent_key_raises(small_spatialdata_path):
    with pytest.raises(KeyError, match="Table 'error' was not found"):
        read_annotated_array(small_spatialdata_path, key="error")



@pytest.fixture
def without_modules(monkeypatch):
    """
    Simulate that some packages are not installed. Setting a `sys.modules` entry
    to `None` makes any subsequent import of that package raise an ImportError.
    The reader module is re-imported so that module-level imports are checked too.
    """
    def wrapper(*module_names):
        for name in module_names:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, "openpipeline_ioutils.reader", raising=False)
        return importlib.import_module("openpipeline_ioutils.reader")

    return wrapper

def test_without_modules_blocks_import(without_modules, small_spatialdata_path):
    # Sanity check: make sure that blocking a module actually causes a failure.
    reader = without_modules("spatialdata")
    with pytest.raises(ImportError):
        reader.read_annotated_array(small_spatialdata_path, key="table1")

def test_read_anndata_without_optional_dependencies(without_modules, small_anndata_1_path):
    reader = without_modules("mudata", "spatialdata")
    output = reader.read_annotated_array(small_anndata_1_path)
    assert isinstance(output, ad.AnnData)

def test_read_mudata_without_spatialdata(without_modules, small_mudata_path):
    reader = without_modules("spatialdata")
    output = reader.read_annotated_array(small_mudata_path, key="mod1")
    assert isinstance(output, ad.AnnData)

def test_read_spatialdata_without_mudata(without_modules, small_spatialdata_path):
    reader = without_modules("mudata")
    output = reader.read_annotated_array(small_spatialdata_path, key="table1")
    assert isinstance(output, ad.AnnData)
