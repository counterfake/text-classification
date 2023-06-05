#!/usr/bin/env python
# coding: utf-8
# %%
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
ZOO_PATH = Path.joinpath(ROOT_PATH, "data", "model_zoo.json")
PROCESSED_DATA_PATH = Path.joinpath(ROOT_PATH, "data", "processed", "data.csv")
MODEL_CV_RESULT_PATH = Path.joinpath(ROOT_PATH, "data", "evaluation")

TARGET_DICT = {"not-risky": 0,
               "risky": 1,
               "second hand": 2}

TARGET_INV_DICT = {TARGET_DICT[key]: key for key in TARGET_DICT.keys()}
