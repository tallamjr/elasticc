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
        print("{} not in history data".format(field))
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


# Order of size:
# 2.2M    classId=121
# 42M     classId=122
# 164M    classId=135
# 239M    classId=123
# 251M    classId=133
# 819M    classId=134
# 887M    classId=124
# 1.2G    classId=132
# 1.3G    classId=211
# 1.5G    classId=114
# 1.5G    classId=115
# 1.8G    classId=131
# 2.6G    classId=213
# 2.7G    classId=112
# 3.2G    classId=221
# 4.5G    classId=214
# 6.2G    classId=111
# 6.6G    classId=113
# 8.0G    classId=212

labels = [
    121,
    # 122,
    # 135,
    # 123,
    # 133,
    # 134,
    # 124,
    # 132,
    # 211,  # TODO: from here
    # 114,
    # 115,
    # 131,
    # 213,
    # 112,
    # 221,
    # 214,  # LARGE
    # 111,  # LARGE
    # 113,  # LARGE
    # 212,  # LARGE
]

sn_like = {111, 112, 113, 114, 115}
fast = {121, 122, 123, 124}
long = {131, 132, 133, 134, 135}
periodic = {211, 212, 213, 214, 215}
non_periodic = {221}

branches = {
    "SN-like": sn_like,
    "Fast": fast,
    "Long": long,
    "Periodic": periodic,
    "Non-Periodic": non_periodic,
}

# cat = "transients"
# cat = "non-transients"
cat = "all-classes"

xfeats = True

# TODO: If label in set, add additional catergory, i.e FAST/RECURRING etc. See taxonomy
for label in labels:

    branch_dict = {k: label in v for k, v in branches.items()}
    branch = [k for k, v in branch_dict.items() if v][0]
    print(f"TAXONOMY BRANCH -- {branch}")

    print(f"PROCESSING classId -- {label} == {CLASS_MAPPING.get(label)}")

    pdf = pl.read_parquet(
        f"{ROOT}/data/raw/ftransfer_elasticc_2023-02-15_946675/classId={label}",
        use_pyarrow=True,
        memory_map=True,
        low_memory=True,
        parallel="columns",
    )

    pdf = pdf.to_pandas()

    pdf["cpsFlux"] = pdf[["diaSource", "prvDiaForcedSources"]].apply(
        lambda x: extract_field(x, "prvDiaForcedSources", "psFlux", "diaSource"),
        axis=1,
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

    cols = [
        "alertId",
        "cmidPointTai",
        "cpsFlux",
        "cpsFluxErr",
        "cfilterName",
        "SNID",
    ]

    additional_features = {}

    # Additional features
    if xfeats:
        pdf["cZ"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
            lambda x: extract_field(x, "prvDiaForcedSources", "z_final", "diaObject"),
            axis=1,
        )
        pdf["cZerr"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
            lambda x: extract_field(x, "prvDiaForcedSources", "z_final_err", "diaObject"),
            axis=1,
        )
        pdf["cMwebv"] = pdf[["diaObject", "prvDiaForcedSources"]].apply(
            lambda x: extract_field(x, "prvDiaForcedSources", "mwebv", "diaObject"),
            axis=1,
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

        additional_features = {
            "cZ": "z",
            "cZerr": "z_error",
            "cMwebv": "mwebv",
            "cRa": "ra",
            "cDecl": "dec",
            "cHostgal_ra": "hostgal_ra",
            "cHostgal_dec": "hostgal_dec",
            "NOBS": "nobs",
        }

        cols.extend(list(additional_features.keys()))

    df = pl.from_pandas(pdf)
    df = df.select(cols)

    sub = (
        df.lazy()
        .filter(pl.col("cfilterName").arr.unique().arr.lengths() > 1)
        .filter(pl.col("cmidPointTai").arr.lengths() > 5)
        .collect(streaming=True)
    )

    if sub.shape[0] == 0:
        print(
            f"ONLY SINGLE BAND, SINGLE DATA POINT OBSERVATIONS. SKIPPING CLASS -- {label}"
        )
        continue

    df = (
        sub.lazy()
        .explode(["cmidPointTai", "cpsFlux", "cpsFluxErr", "cfilterName"])
        .sort(by="cfilterName")
    )

    df = df.rename(
        {
            "alertId": "object_id",
            "SNID": "uuid",
            "cmidPointTai": "mjd",
            "cpsFlux": "flux",
            "cpsFluxErr": "flux_error",
            "cfilterName": "filter",
        }
    )

    if xfeats:
        df = df.explode(
            [x for x in list(additional_features.keys()) if (x != "SNID" and x != "NOBS")]
            # ["cZ", "cZerr", "cMwebv", "cRa", "cDecl", "cHostgal_ra", "cHostgal_dec"]
        )
        df = df.rename(additional_features)

    # TODO: make polars version of remap_filters.
    # df = remap_filters(df, filter_map=ELASTICC_FILTER_MAP)
    # df = df.rename({"passband": "filter"})
    df = df.with_columns(pl.col("filter").map_dict(ELASTICC_FILTER_MAP)).collect(
        streaming=True
    )

    assert df.shape[1] == (
        len(
            [
                "object_id",
                "mjd",
                "flux",
                "flux_error",
                "filter",
                "uuid",
            ]
        )
        + len(additional_features)
    )

    alert_list = list(np.unique(df["object_id"]))
    print(f"NUM ALERTS TO BE PROCESSED: {len(alert_list)}")

    generated_gp_dataset = generate_gp_all_objects(alert_list, df)

    generated_gp_dataset = (
        generated_gp_dataset.lazy()
        .with_columns([pl.lit(label).alias("target"), pl.lit(branch).alias("branch")])
        .collect(streaming=True)
    )

    generated_gp_dataset.write_parquet(
        f"{ROOT}/data/processed/{cat}/gps-classId-{label}.parquet"
    )

    assert generated_gp_dataset.select("object_id").unique().height == len(alert_list)
    # print(generated_gp_dataset)

    time_series_feats = [
        "mjd",
        "lsstg",
        "lssti",
        "lsstr",
        "lsstu",
        "lssty",
        "lsstz",
    ]

    assert generated_gp_dataset.shape == (
        len(alert_list) * 100,
        len(time_series_feats) + len(["object_id", "target", "branch"]),
    )

    df_merge = df.lazy().drop(columns=["mjd", "filter", "flux", "flux_error"]).collect()
    df_merge.write_parquet(f"{ROOT}/data/processed/{cat}/xfeats-classId-{label}.parquet")

    # df_semi = generated_gp_dataset.lazy().join(
    #     df_merge.lazy(), on="object_id", how="semi"
    # )

    # df_with_xfeats = (
    #     df_semi.lazy()
    #     .join(df_merge.lazy(), on="object_id", how="inner")
    #     .unique()
    #     .collect()
    # )

    df_with_xfeats = generated_gp_dataset.join(
        df_merge, on="object_id", how="inner"
    ).unique()

    # df_with_xfeats.drop_duplicates(keep="first", inplace=True, ignore_index=True)

    assert df_with_xfeats.shape == (
        len(alert_list) * 100,
        len(time_series_feats)
        + len(additional_features)
        + len(["object_id", "target", "uuid", "branch"]),
    )

    # df_with_xfeats = df_with_xfeats.lazy().with_columns(pl.lit(branch).alias("branch"))
    # print(df_with_xfeats)

    # change dtypes for maximal file compression
    # pldf = pl.from_pandas(df_with_xfeats)
    pldf = df_with_xfeats.lazy().with_columns(
        [
            pl.all().cast(pl.Float32, strict=False),
            pl.col("object_id").cast(pl.UInt64, strict=False),
            pl.col("uuid").cast(pl.UInt32, strict=False),
            pl.col("target").cast(pl.UInt8, strict=False),
            pl.col("branch").cast(pl.Utf8, strict=False),
        ]
    )

    pldf.sink_parquet(f"{ROOT}/data/processed/{cat}/classId-{label}.parquet")

    print(pldf.head().collect())

    # WIP: Test functions
    # test viz function
    viz_num_filters = 6
    while viz_num_filters == 6:
        data = df.filter(pl.col("object_id") == random.choice(alert_list))
        # data = df[df["object_id"] == random.choice(alert_list)]
        _obj_gps = generate_gp_single_event(data)
        ax = plot_event_data_with_model(
            data.to_pandas(), obj_model=_obj_gps.to_pandas(), pb_colors=ELASTICC_PB_COLORS
        )
        viz_num_filters = len(data["filter"].unique())
        print(f"RAN VIZ TEST WITH {viz_num_filters} FILTERS")

    del (
        _obj_gps,
        alert_list,
        ax,
        data,
        df,
        df_merge,
        df_with_xfeats,
        generated_gp_dataset,
        pdf,
        pldf,
        sub,
        viz_num_filters,
    )
    gc.collect()
