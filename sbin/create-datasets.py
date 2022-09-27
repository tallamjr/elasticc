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

import glob

# Data Processing Pipeline of 'training_alerts'
# Refs: https://zenodo.org/record/7017557#.YyrF-uzMKrM
import pprint
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from astronet.preprocess import one_hot_encode
from astronet.utils import create_dataset
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

queries = []
for file in glob.glob("../data/processed/t2/*.xz"):
    q = pd.read_csv(file)
    queries.append(q)

df = pd.concat(queries)

print(df.head())
print(df.shape)

x = [
    "lsstg",
    "lssti",
    "lsstr",
    "lsstu",
    "lssty",
    "lsstz",
]

Xs, ys = create_dataset(df[x], df.target, time_steps=100, step=100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    Xs, ys, stratify=ys, random_state=RANDOM_SEED
)

z = [
    "hostgal_photoz",
    "hostgal_photoz_err",
]

Zs, ys = create_dataset(df[z], df.target, time_steps=100, step=100)
Z_train, Z_test, y_train, y_test = model_selection.train_test_split(
    Zs, ys, stratify=ys, random_state=RANDOM_SEED
)

Z_train = np.mean(Z_train, axis=1)
Z_test = np.mean(Z_test, axis=1)

scaler = RobustScaler()
X_train = X_train.reshape(X_train.shape[0] * 100, 6)
X_train = scaler.fit(X_train).transform(X_train)
X_train = X_train.reshape(X_train.shape[0] // 100, 100, 6)

scaler = RobustScaler()
X_test = X_test.reshape(X_test.shape[0] * 100, 6)
X_test = scaler.fit(X_test).transform(X_test)
X_test = X_test.reshape(X_test.shape[0] // 100, 100, 6)

scaler = RobustScaler()
Z_train = scaler.fit(Z_train).transform(Z_train)

scaler = RobustScaler()
Z_test = scaler.fit(Z_test).transform(Z_test)

pprint.pprint(Counter(y_train.squeeze()))
pprint.pprint(Counter(y_test.squeeze()))
assert set(np.unique(y_train)) == set(np.unique(y_test))

# One hot encode y
enc, y_train, y_test = one_hot_encode(y_train, y_test)
encoding_file = "../data/processed/t2/dataset.enc"

with open(encoding_file, "wb") as f:
    joblib.dump(enc, f)

print("SAVING NEW DATASET")
# passbands
np.save("../data/processed/t2/X_train.npy", X_train)
np.save("../data/processed/t2/X_test.npy", X_test)

# redshift
np.save("../data/processed/t2/Z_train.npy", Z_train)
np.save("../data/processed/t2/Z_test.npy", Z_test)

# labels
np.save("../data/processed/t2/y_train.npy", y_train)
np.save("../data/processed/t2/y_test.npy", y_test)

print(
    f"TRAIN SHAPES:\n x = {X_train.shape} \n z = {Z_train.shape} \n y = {y_train.shape}"
)
print(
    f"TEST SHAPES:\n x = {X_test.shape} \n z = {Z_test.shape} \n y = {y_test.shape} \n"
)
