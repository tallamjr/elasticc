# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Data Processing Pipeline of 'training_alerts'
# Refs: https://zenodo.org/record/7017557#.YyrF-uzMKrM

import gc
import glob

import numpy as np
import polars as pl
from astronet.constants import (
    ELASTICC_FILTER_MAP,
    ELASTICC_PB_COLORS,
    ELASTICC_PB_WAVELENGTHS,
)
from astronet.preprocess import generate_gp_all_objects, remap_filters
from astronet.viz.visualise_data import plot_event_gp_mean

classes = [
    # "AGN",
    # "CART",
    # "Cepheid",
    # "EB", # EB_3.tar ERROR
    "ILOT",
    "KN_B19",
    "KN_K17",
    # "Mdwarf-flare",
    "PISN",
    "RRL",
    "SLSN-I+host",
    "SLSN-I_no_host",
    "SNII+HostXT_V19",
    "SNII-NMF",
    "SNII-Templates",
    "SNIIb+HostXT_V19",
    "SNIIn+HostXT_V19",
    "SNIIn-MOSFIT",
    "SNIa-91bg",
    "SNIa-SALT2",
    "SNIax",
    "SNIb+HostXT_V19",
    "SNIb-Templates",
    "SNIc+HostXT_V19",
    "SNIc-Templates",
    "SNIcBL+HostXT_V19",
    "TDE",
    "d-Sct",
    # "dwarf-nova", # EMPTY DIR
    "uLens-Binary",
    "uLens-Single-GenLens",
    "uLens-Single_PyLIMA",
]

for transient in classes:
    queries = []
    for file in glob.glob(f"../data/processed/training_alerts_v3/{transient}/*"):
        q = pl.scan_csv(file)
        queries.append(q)

    dataframes = pl.collect_all(queries)

    pldf = pl.concat(dataframes)

    pdf = pldf.to_pandas()

    pdf.rename(
        {
            "HOSTGAL_PHOTOZ": "hostgal_photoz",
            "HOSTGAL_PHOTOZ_ERR": "hostgal_photoz_err",
            "FLUXCAL": "flux",
            "FLUXCALERR": "flux_error",
            "MJD": "mjd",
            "candid": "object_id",
            "type": "target",
            "BAND": "passband",
        },
        axis="columns",
        inplace=True,
    )

    df = pdf.filter(
        items=[
            "mjd",
            "flux",
            "flux_error",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "passband",
            "object_id",
            "target",
        ]
    )

    df = remap_filters(df, filter_map=ELASTICC_FILTER_MAP)

    object_list = list(np.unique(df["object_id"]))
    print(f"NUM TOTAL ALERTS FOR {transient}: {len(object_list)}")

    generated_gp_dataset = generate_gp_all_objects(
        object_list,
        df,
        timesteps=100,
        pb_wavelengths=ELASTICC_PB_WAVELENGTHS,
    )
    generated_gp_dataset["object_id"] = generated_gp_dataset["object_id"].astype(int)

    print(
        f"NUM PROCESSED ALERTS FOR {transient}: {len(np.unique(generated_gp_dataset['object_id']))}"
    )

    ddf = df[["hostgal_photoz", "hostgal_photoz_err", "object_id", "target"]]

    dfwz = generated_gp_dataset.merge(ddf, on="object_id", how="left").drop_duplicates()

    dfwz.to_csv(
        f"../data/processed/t2/{transient}.xz", compression="infer", index=False
    )

    # plot_event_gp_mean(
    #     generated_gp_dataset,
    #     object_id=dfwz["object_id"][0],
    #     pb_colors=ELASTICC_PB_COLORS,
    # )

    f, ax = plot_event_gp_mean(
        dfwz.filter(
            items=[
                "mjd",
                "lsstg",
                "lssti",
                "lsstr",
                "lsstu",
                "lssty",
                "lsstz",
                "object_id",
            ]
        ),
        object_id=dfwz["object_id"][0],
        pb_colors=ELASTICC_PB_COLORS,
    )

    f.savefig(f"../data/processed/t2/{transient}.pdf")

    del [[pldf, pdf, df, generated_gp_dataset, ddf, dfwz]]
    gc.collect()
