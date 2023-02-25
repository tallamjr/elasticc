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
from elasticc.constants import ROOT
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

cat = "transients"
# cat  = "non-transients"
# cat = "all-labels"

df = pl.scan_parquet(f"{ROOT}/data/processed/{cat}/classId-1*.parquet")
df.sink_parquet(f"{ROOT}/data/processed/train.parquet")

df = pl.read_parquet(f"{ROOT}/data/processed/train.parquet")
# df = pl.read_parquet(f"{ROOT}/data/processed/training-transient/classId-121-*")

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
    df[x], df["target"], df["uuid"], time_steps=num_gps, step=num_gps
)

print(groups.shape)

# gss = model_selection.StratifiedGroupKFold(n_splits=2)
gss = model_selection.GroupShuffleSplit(
    n_splits=1, random_state=RANDOM_SEED, test_size=None, train_size=0.7
)
gss.get_n_splits()

print(gss)

for i, (train_index, test_index) in enumerate(gss.split(Xs, ys, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    import pdb

    pdb.set_trace()


np.save(f"{ROOT}/data/processed/groups.npy", groups)
np.save(f"{ROOT}/data/processed/groups_train_idx.npy", groups[train_index])

X_train = Xs[train_index]
X_test = Xs[test_index]

y_train = ys[train_index]
y_test = ys[test_index]

scaler = RobustScaler()
X_train = X_train.reshape(X_train.shape[0] * 100, 6)
X_train = scaler.fit(X_train).transform(X_train)
X_train = X_train.reshape(X_train.shape[0] // 100, 100, 6)

scaler = RobustScaler()
X_test = X_test.reshape(X_test.shape[0] * 100, 6)
X_test = scaler.fit(X_test).transform(X_test)
X_test = X_test.reshape(X_test.shape[0] // 100, 100, 6)

pprint.pprint(Counter(y_train.squeeze()))
pprint.pprint(Counter(y_test.squeeze()))
assert set(np.unique(y_train)) == set(np.unique(y_test))

# One hot encode y
enc, y_train, y_test = one_hot_encode(y_train, y_test)
encoding_file = f"{ROOT}/data/processed/transient-dataset.enc"

with open(encoding_file, "wb") as f:
    joblib.dump(enc, f)

print("SAVING NEW DATASET")

# passbands
np.save(f"{ROOT}/data/processed/X_train.npy", X_train)
np.save(f"{ROOT}/data/processed/X_test.npy", X_test)

# labels
np.save(f"{ROOT}/data/processed/y_train.npy", y_train)
np.save(f"{ROOT}/data/processed/y_test.npy", y_test)

z = [
    # "z",
    # "z_error",
]

if z:

    Zs, ys, _ = create_dataset(df[z], df["target"], df["uuid"], time_steps=100, step=100)

    Z_train = Zs[train_index]
    Z_test = Zs[test_index]

    Z_train = np.mean(Z_train, axis=1)
    Z_test = np.mean(Z_test, axis=1)

    scaler = RobustScaler()
    Z_train = scaler.fit(Z_train).transform(Z_train)

    scaler = RobustScaler()
    Z_test = scaler.fit(Z_test).transform(Z_test)

    # z = [
    #     "ra",
    #     "dec",
    #     "hostgal_ra",
    #     "hostgal_dec",
    #     "nobs",
    # ]

    # Zs, ys, _ = create_dataset(df[z], df["target"], df["uuid"], time_steps=100, step=100)

    # Z_train_add = Zs[train_index]
    # Z_test_add = Zs[test_index]

    # Z_train_add = np.mean(Z_train_add, axis=1)
    # Z_test_add = np.mean(Z_test_add, axis=1)

    # Z_train = np.hstack((Z_train, Z_train_add))
    # Z_test = np.hstack((Z_test, Z_test_add))

    # additional features
    np.save(f"{ROOT}/data/processed/Z_train.npy", Z_train)
    np.save(f"{ROOT}/data/processed/Z_test.npy", Z_test)

    print(
        f"TRAIN SHAPES:\n x = {X_train.shape} \n z = {Z_train.shape} \n y = {y_train.shape}"
    )
    print(
        f"TEST SHAPES:\n x = {X_test.shape} \n z = {Z_test.shape} \n y = {y_test.shape} \n"
    )
else:
    print(f"TRAIN SHAPES:\n x = {X_train.shape} \n y = {y_train.shape}")
    print(f"TEST SHAPES:\n x = {X_test.shape} \n y = {y_test.shape} \n")
