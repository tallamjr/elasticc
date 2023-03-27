import gc
import random

import numpy as np
import pandas as pd
import polars as pl

# from astronet.constants import LSST_FILTER_MAP as ELASTICC_FILTER_MAP
from astronet.constants import ELASTICC_FILTER_MAP
from astronet.constants import LSST_PB_COLORS as ELASTICC_PB_COLORS
from astronet.preprocess import (
    generate_gp_all_objects,
    generate_gp_single_event,
    remap_filters,
)
from astronet.viz.visualise_data import plot_event_data_with_model
from elasticc.constants import CLASS_MAPPING, ROOT

SEED = 9001

pd.options.mode.dtype_backend = "pyarrow"

df = (
    pl.scan_parquet(f"{ROOT}/data/processed/all-classes-tsonly/class*")
    .with_columns(pl.col("target").cast(pl.Int64).map_dict(CLASS_MAPPING))
    .limit(10_000)
    .collect(streaming=True)
)
import pdb

pdb.set_trace()

data = df.filter(pl.col("object_id") == random.choice(alert_list))
_obj_gps = generate_gp_single_event(data)
ax = plot_event_data_with_model(
    data.to_pandas(), obj_model=_obj_gps.to_pandas(), pb_colors=ELASTICC_PB_COLORS
)
viz_num_filters = len(data["filter"].unique())
