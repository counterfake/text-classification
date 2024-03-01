#!/usr/bin/env python
# coding: utf-8
# %%
from pathlib import Path
import json

ROOT_PATH = Path(__file__).parent.parent.parent
ZOO_PATH = Path.joinpath(ROOT_PATH, "data", "model_zoo.json")
INDICATOR_WORDS = json.load(open(f"{ROOT_PATH}/src/utils/words.json", "r"))
MODEL_CV_RESULT_PATH = Path.joinpath(ROOT_PATH, "data", "evaluation")

TARGET_DICT_FASHION = {
    1001: 0,
    1101: 1,
    1201: 2,
    1202: 3,
    1203: 4,
    1301: 5,
    1302: 6,
    1401: 7,
    1501: 8,
    1502: 9,
    1701: 10,
    1702: 11,
    1801: 12,
    201: 13,
    202: 14,
    203: 15,
    204: 16,
    205: 17,
    206: 18,
    301: 19,
    302: 20,
    303: 21,
    401: 22,
    402: 23,
    501: 24,
    502: 25,
    503: 26,
    601: 27,
    602: 28,
    603: 29,
    604: 30,
    605: 31,
    701: 32,
    702: 33,
    703: 34,
    704: 35,
    705: 36,
    706: 37,
    707: 38,
    708: 39,
    709: 40,
    710: 41,
    711: 42,
    712: 43,
    801: 44,
    901: 45,
    902: 46,
    903: 47,
    904: 48,
}
TARGET_INV_DICT_FASHION = {
    TARGET_DICT_FASHION[key]: key for key in TARGET_DICT_FASHION.keys()
}

TARGET_DICT = {"0.NotRisky": 0, "1.Risky": 1, "2.SecondHand": 2}
TARGET_INV_DICT = {TARGET_DICT[key]: key for key in TARGET_DICT.keys()}
