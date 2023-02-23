import gc
import random

import numpy as np
import pandas as pd
import polars as pl
from astronet.constants import ELASTICC_FILTER_MAP, ELASTICC_PB_COLORS
from astronet.preprocess import (
    generate_gp_all_objects,
    generate_gp_single_event,
    remap_filters,
)
from astronet.viz.visualise_data import plot_event_data_with_model
from elasticc.constants import ROOT

SEED = 9001


def extract_history(history_list: list, field: str) -> list:
    """Extract the historical measurements contained in the alerts
    for the parameter `field`.

    Parameters
    ----------
    history_list: list of dict
        List of dictionary from alert[history].
    field: str
        The field name for which you want to extract the data. It must be
        a key of elements of history_list

    Returns
    ----------
    measurement: list
        List of all the `field` measurements contained in the alerts.
    """
    if history_list is None:
        return []
    try:
        measurement = [obs[field] for obs in history_list]
    except KeyError:
        # print("{} not in history data".format(field))
        measurement = []

    return measurement


def extract_field(alert: dict, category: str, field: str, key: str) -> np.array:
    """Concatenate current and historical observation data for a given field.

    Parameters
    ----------
    alert: dict
        Dictionnary containing alert data
    category: str
        prvDiaSources or prvDiaForcedSources
    field: str
        Name of the field to extract.

    Returns
    ----------
    data: np.array
        List containing previous measurements and current measurement at the
        end. If `field` is not in the category, data will be
        [alert['diaSource'][field]].
    """
    data = np.concatenate([[alert[key][field]], extract_history(alert[category], field)])
    return data


labels = [
    # 111,  # LARGE 🚧
    # 112, # ✅
    # 113,  # LARGE 🚧
    # 114, # ✅
    # 115, # ✅
    # 121, # ✅
    # 122, # ✅
    # 123, # ✅
    # 124, # ✅
    # 131, # ✅
    # 132, # ✅
    # 133, # ✅
    # 134, # ✅
    # 135, # ✅
    # 211, # ✅
    # 212,  # LARGE 🚧
    # 213, # ✅
    214,  # LARGE 🚧
    # 221, # ✅
]

for label in labels:

    print(f"PROCESSING classId -- {label}")

    df = pd.read_parquet(
        f"{ROOT}/data/raw/ftransfer_elasticc_2023-02-15_946675/classId={label}"
    )
    if df.shape[0] >= 1000000:  # if dataframe greater than a million rows
        pdf = df.sample(frac=0.2, random_state=SEED)  # then reduce down to 20%
        del df
        gc.collect()

    pdf["target"] = label

    pdf["cpsFlux"] = pdf[["diaSource", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "psFlux", "diaSource"), axis=1
    )
    pdf["cpsFluxErr"] = pdf[["diaSource", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "psFluxErr", "diaSource"),
        axis=1,
    )
    pdf["cfilterName"] = pdf[["diaSource", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "filterName", "diaSource"),
        axis=1,
    )
    pdf["cmidPointTai"] = pdf[["diaSource", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "midPointTai", "diaSource"),
        axis=1,
    )

    # custom, additional features = [
    # "z_final",
    # "z_final_err",
    # "mwebv",
    # "ra",
    # "dec",
    # "hostgal_ra",
    # "hostgal_dec",
    # "NOBS",
    # ]
    pdf["cZ"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "z_final", "diaObject"), axis=1
    )
    pdf["cZerr"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "z_final_err", "diaObject"),
        axis=1,
    )
    pdf["cMwebv"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "mwebv", "diaObject"), axis=1
    )
    pdf["cRa"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "ra", "diaObject"), axis=1
    )
    pdf["cDecl"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "decl", "diaObject"), axis=1
    )
    pdf["cHostgal_ra"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "hostgal_ra", "diaObject"),
        axis=1,
    )
    pdf["cHostgal_dec"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "hostgal_dec", "diaObject"),
        axis=1,
    )

    cols = [
        "alertId",
        "target",
        "cmidPointTai",
        "cpsFlux",
        "cpsFluxErr",
        "cfilterName",
        "cZ",
        "cZerr",
        "cMwebv",
        "cRa",
        "cDecl",
        "cHostgal_ra",
        "cHostgal_dec",
        "SNID",
        "NOBS",
    ]
    sub = pdf[cols]

    def f(x):
        y = len(list(set(x))) > 1
        # Require at least observations across more than one passband
        return y

    df = sub["cfilterName"].apply(f)
    sub = sub[df].reset_index()

    def f(x):
        y = len(x) > 1
        # Keep rows with more than 12 data points
        return y

    df = sub["cmidPointTai"].apply(f)
    sub = sub[df].reset_index(drop=True)

    if sub.shape[0] == 0:
        print(
            f"ONLY SINGLE BAND, SINGLE DATA POINT OBSERVATIONS. SKIPPING CLASS -- {label}"
        )
        continue

    df = sub.explode(
        column=["cmidPointTai", "cpsFlux", "cpsFluxErr", "cfilterName"]
    ).sort_values(by=["index", "cfilterName"])

    df = df.explode(
        column=["cZ", "cZerr", "cMwebv", "cRa", "cDecl", "cHostgal_ra", "cHostgal_dec"]
    )

    df = df.rename(
        columns={
            "alertId": "object_id",
            "SNID": "uuid",
            "cmidPointTai": "mjd",
            "cpsFlux": "flux",
            "cpsFluxErr": "flux_error",
            "cfilterName": "filter",
            "cZ": "z",
            "cZerr": "z_error",
            "cMwebv": "mwebv",
            "cRa": "ra",
            "cDecl": "dec",
            "cHostgal_ra": "hostgal_ra",
            "cHostgal_dec": "hostgal_dec",
            "NOBS": "nobs",
        }
    )

    df = remap_filters(df, filter_map=ELASTICC_FILTER_MAP)

    assert df.shape[1] == 16

    df = df.drop(columns=["index"])

    alert_list = list(np.unique(df["object_id"]))
    print(f"NUM ALERTS TO BE PROCESSED: {len(alert_list)}")

    if len(alert_list) > 100000:  # if num alerts > 100,000
        print(f"SUB-SAMPPLING NUM OBJECT LIST FROM {len(alert_list)} TO 10000")
        random.seed(SEED)
        alert_list = random.sample(alert_list, 10000)  # then sub-sample down to 10, 000

    if len(alert_list) >= 5000:
        chunk_list = np.array_split(alert_list, 100)
    else:
        chunk_list = np.array_split(alert_list, 1)

    for num, chunk in enumerate(chunk_list):

        print(f"ITERATION : {num}")

        ddf = df[df["object_id"].isin(chunk_list[num])]
        print(f"NUM ALERTS IN CHUNK : {len(chunk)}")
        generated_gp_dataset = generate_gp_all_objects(chunk, ddf)
        generated_gp_dataset["target"] = label

        assert len(generated_gp_dataset["object_id"].unique()) == len(chunk)
        print(generated_gp_dataset)
        assert generated_gp_dataset.shape == (len(chunk) * 100, 9)

        df_merge = df.drop(columns=["mjd", "filter", "flux", "flux_error", "target"])
        df_with_xfeats = generated_gp_dataset.merge(df_merge, on="object_id", how="inner")
        df_with_xfeats.drop_duplicates(keep="first", inplace=True, ignore_index=True)
        assert df_with_xfeats.shape == (len(chunk) * 100, 18)

        # change dtypes for maximal file compression
        pldf = pl.from_pandas(df_with_xfeats)
        pldf = pldf.with_columns(
            [
                pl.all().cast(pl.Float32, strict=False),
                pl.col("object_id").cast(pl.UInt64, strict=False),
                pl.col("uuid").cast(pl.UInt32, strict=False),
                pl.col("target").cast(pl.UInt8, strict=False),
                pl.col("nobs").cast(pl.UInt8, strict=False),
            ]
        )

        pldf.write_parquet(
            f"{ROOT}/data/processed/training-transient/classId-{label}-{num:03d}.parquet"
        )

        del (
            ddf,
            df_merge,
            df_with_xfeats,
            generated_gp_dataset,
        )
        gc.collect()

    # test viz function
    viz_num_filters = 6
    while viz_num_filters == 6:
        data = df[df["object_id"] == random.choice(alert_list)]
        _obj_gps = generate_gp_single_event(data)
        ax = plot_event_data_with_model(
            data, obj_model=_obj_gps, pb_colors=ELASTICC_PB_COLORS
        )
        viz_num_filters = len(data["filter"].unique())
        print(f"RAN VIZ TEST WITH {viz_num_filters} FILTERS")

    del (
        _obj_gps,
        alert_list,
        ax,
        data,
        df,
        pdf,
        pldf,
        sub,
        viz_num_filters,
    )
    gc.collect()
