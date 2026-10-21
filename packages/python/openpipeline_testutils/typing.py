from pathlib import Path
from typing import Union

from anndata import AnnData
from mudata import MuData

AnnotationObject = Union[MuData, AnnData]
AnnotationObjectOrPathLike = Union[AnnotationObject, str, Path]
