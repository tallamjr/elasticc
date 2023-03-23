# Copyright 2020 - 2023
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
import pprint
from collections import Counter

import joblib
import numpy as np
import polars as pl
from astronet.preprocess import one_hot_encode
from astronet.utils import create_dataset
from elasticc.constants import CLASS_MAPPING, ROOT
from imblearn.under_sampling import RandomUnderSampler
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

cat = "all-classes"

xfeats = False  # Incluce additional features? This will reduce number of possible alerts
# expensive_join = True  # Run expensive join as part of this script or separaetly

cat = cat + "-xfeats" if xfeats else cat + "-tsonly"

# df = pl.scan_parquet(f"{ROOT}/data/processed/{cat}/classId-1*.parquet")
# df.sink_parquet(f"{ROOT}/data/processed/train.parquet")

# df = pl.read_parquet(f"{ROOT}/data/processed/train.parquet")
df = pl.scan_parquet(f"{ROOT}/data/processed/{cat}/class*.parquet").with_columns(
    pl.col("target").cast(pl.Int64).map_dict(CLASS_MAPPING)
)

# df = pl.read_parquet(f"{ROOT}/data/processed/training-transient/classId-121-*")

# df.with_columns(pl.col("target").cast(pl.Int64).map_dict(CLASS_MAPPING))
# shape: (27942800, 10)
# ┌──────────────┬────────────┬─────────────┬─────────────┬─────┬─────────────┬──────────────┬────────┬───────────┐
# │ mjd          ┆ lsstg      ┆ lssti       ┆ lsstr       ┆ ... ┆ lsstz       ┆ object_id    ┆ target ┆ uuid      │
# │ ---          ┆ ---        ┆ ---         ┆ ---         ┆     ┆ ---         ┆ ---          ┆ ---    ┆ ---       │
# │ f32          ┆ f32        ┆ f32         ┆ f32         ┆     ┆ f32         ┆ u64          ┆ str    ┆ u32       │
# ╞══════════════╪════════════╪═════════════╪═════════════╪═════╪═════════════╪══════════════╪════════╪═══════════╡
# │ 60378.34375  ┆ -71.350517 ┆ 224.562103  ┆ 188.895645  ┆ ... ┆ -28.738251  ┆ 16656038043  ┆ SNIa   ┆ 8328019   │
# │ 60378.65625  ┆ -71.349266 ┆ 230.952515  ┆ 191.703491  ┆ ... ┆ -23.553474  ┆ 16656038043  ┆ SNIa   ┆ 8328019   │
# │ 60378.96875  ┆ -71.243782 ┆ 237.522507  ┆ 194.647873  ┆ ... ┆ -18.159994  ┆ 16656038043  ┆ SNIa   ┆ 8328019   │
# │ 60379.28125  ┆ -71.029121 ┆ 244.291138  ┆ 197.786713  ┆ ... ┆ -12.550078  ┆ 16656038043  ┆ SNIa   ┆ 8328019   │
# │ ...          ┆ ...        ┆ ...         ┆ ...         ┆ ... ┆ ...         ┆ ...          ┆ ...    ┆ ...       │
# │ 60632.960938 ┆ 280.735565 ┆ 2707.381104 ┆ 1436.50769  ┆ ... ┆ 3839.870605 ┆ 310189040074 ┆ PISN   ┆ 155094520 │
# │ 60634.988281 ┆ 274.275299 ┆ 2706.633789 ┆ 1430.845215 ┆ ... ┆ 3841.236816 ┆ 310189040074 ┆ PISN   ┆ 155094520 │
# │ 60637.015625 ┆ 268.01236  ┆ 2705.712891 ┆ 1425.199585 ┆ ... ┆ 3842.293945 ┆ 310189040074 ┆ PISN   ┆ 155094520 │
# │ 60639.039062 ┆ 261.942627 ┆ 2704.630127 ┆ 1419.590454 ┆ ... ┆ 3843.044922 ┆ 310189040074 ┆ PISN   ┆ 155094520 │
# └──────────────┴────────────┴─────────────┴─────────────┴─────┴─────────────┴──────────────┴────────┴───────────┘

df = df.collect()
# df = df.limit(10000).collect()
# df = df.with_columns([pl.col(x).shift(num_gps).alias(f"A_lag_{i}") for i in range(df.height)]).select([pl.concat_list([f"A_lag_{i}" for i in range(num_gps)][::-1]).alias("A_rolling")])

# Xs, ys, groups = create_dataset(
#     df.select(pl.col(x)).collect(),
#     df.select(pl.col("target")).collect(),
#     df.select(pl.col("uuid")).collect(),
#     time_steps=num_gps,
#     step=num_gps,
# )

x = [
    "lsstg",
    "lssti",
    "lsstr",
    "lsstu",
    "lssty",
    "lsstz",
]

num_gps = 100

Xs, ys, groups = create_dataset(
    df.select(x), df.select("target"), df.select("uuid"), time_steps=num_gps, step=num_gps
)

print(groups.shape)

# gss = model_selection.StratifiedGroupKFold(n_splits=2)
gss = model_selection.GroupShuffleSplit(
    n_splits=1, random_state=RANDOM_SEED, test_size=None, train_size=0.8
)
gss.get_n_splits()

print(gss)

for i, (train_index, test_index) in enumerate(gss.split(Xs, ys, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

np.save(f"{ROOT}/data/processed/groups.npy", groups)
np.save(f"{ROOT}/data/processed/groups_train_idx.npy", groups[train_index])

X_train = Xs[train_index]
X_test = Xs[test_index]

y_train = ys[train_index]
y_test = ys[test_index]

# X_train, X_test, y_train, y_test = model_selection.train_test_split(Xs, ys, stratify=ys)

scaler = RobustScaler()
X_train = X_train.reshape(X_train.shape[0] * 100, 6)
X_train = scaler.fit(X_train).transform(X_train)
X_train = X_train.reshape(X_train.shape[0] // 100, 100, 6)

scaler = RobustScaler()
X_test = X_test.reshape(X_test.shape[0] * 100, 6)
X_test = scaler.fit(X_test).transform(X_test)
X_test = X_test.reshape(X_test.shape[0] // 100, 100, 6)

# # sampler = SVMSMOTE(sampling_strategy="not majority")
# # sampler = InstanceHardnessThreshold(sampling_strategy="not minority")
# sampler = RandomUnderSampler(sampling_strategy="not minority")

# X_resampled, y_resampled = sampler.fit_resample(
#     X_train.reshape(X_train.shape[0], -1), y_train
# )

# # Re-shape 2D data back to 3D original shape, i.e (BATCH_SIZE, timesteps, num_features)
# X_resampled = np.reshape(X_resampled, (X_resampled.shape[0], 100, 6))

# X_train = X_resampled
# y_train = y_resampled

pprint.pprint(Counter(y_train.squeeze()))
pprint.pprint(Counter(y_test.squeeze()))
assert set(np.unique(y_train)) == set(np.unique(y_test))

# One hot encode y
enc, y_train, y_test = one_hot_encode(y_train, y_test)
encoding_file = f"{ROOT}/data/processed/{cat}/labels.enc"

with open(encoding_file, "wb") as f:
    joblib.dump(enc, f)

print("SAVING NEW DATASET")

# passbands
np.save(f"{ROOT}/data/processed/{cat}/X_train.npy", X_train)
np.save(f"{ROOT}/data/processed/{cat}/X_test.npy", X_test)

# labels
np.save(f"{ROOT}/data/processed/{cat}/y_train.npy", y_train)
np.save(f"{ROOT}/data/processed/{cat}/y_test.npy", y_test)

if xfeats:

    z = ["z", "z_error"]

    # redshift
    Zs, ys, _ = create_dataset(
        df.select(["z", "z_error"]),
        df.select("target"),
        df.select("uuid"),
        time_steps=num_gps,
        step=100,
    )

    Z_train = Zs[train_index]
    Z_test = Zs[test_index]

    Z_train = np.mean(Z_train, axis=1)
    Z_test = np.mean(Z_test, axis=1)

    scaler = RobustScaler()
    Z_train = scaler.fit(Z_train).transform(Z_train)

    scaler = RobustScaler()
    Z_test = scaler.fit(Z_test).transform(Z_test)

    # other feats
    zplus = [
        "ra",
        "dec",
        "hostgal_ra",
        "hostgal_dec",
        "nobs",
    ]

    if zplus:

        Zs, ys, _ = create_dataset(
            df.select(zplus),
            df.select("target"),
            df.select("uuid"),
            time_steps=100,
            step=100,
        )

        Z_train_add = Zs[train_index]
        Z_test_add = Zs[test_index]

        Z_train_add = np.mean(Z_train_add, axis=1)
        Z_test_add = np.mean(Z_test_add, axis=1)

        Z_train = np.hstack((Z_train, Z_train_add))
        Z_test = np.hstack((Z_test, Z_test_add))

        z.extend(zplus)

    xfeatures = "_".join(z)

    # additional features
    np.save(f"{ROOT}/data/processed/{cat}/Z_train_{xfeatures}.npy", Z_train)
    np.save(f"{ROOT}/data/processed/{cat}/Z_test_{xfeatures}.npy", Z_test)

    print(
        f"TRAIN SHAPES:\n x = {X_train.shape} \n z = {Z_train.shape} \n y = {y_train.shape}"
    )
    print(
        f"TEST SHAPES:\n x = {X_test.shape} \n z = {Z_test.shape} \n y = {y_test.shape} \n"
    )
else:
    print(f"TRAIN SHAPES:\n x = {X_train.shape} \n y = {y_train.shape}")
    print(f"TEST SHAPES:\n x = {X_test.shape} \n y = {y_test.shape} \n")
