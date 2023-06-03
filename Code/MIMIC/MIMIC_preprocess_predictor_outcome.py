#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiaguo
"""

# import Modules
import os
import math
import timeit
import compress_pickle
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

import pyreadr
import random

work_data = compress_pickle.load("Data/MIMIC_seq_data_unique.gz")
subj_count = work_data.groupby('SUBJECT_ID').apply(lambda x: len(x)).reset_index()
subj_count = subj_count.loc[subj_count.iloc[:,1]>1]
Counter(subj_count.iloc[:,1])
work_data = work_data.loc[work_data.SUBJECT_ID.isin(subj_count.SUBJECT_ID)]

""" sequence data as predictor """
work_data['time_rank'] = work_data.groupby(['SUBJECT_ID']).ADMITTIME.rank()
work_data = work_data.sort_values(['SUBJECT_ID', 'time_rank']).reset_index(drop=True)

idx = list(work_data.groupby(['SUBJECT_ID']).time_rank.idxmax())
temp = work_data.drop(columns=['time_rank', 'death', 'event']).iloc[idx,:].reset_index(drop=True)
temp = temp.rename(columns={"ADMITTIME":"outcome_time", "HADM_ID":"outcome_HADM"})
work_data = pd.merge(work_data, temp, how='left')

work_data['diff_time'] = work_data.outcome_time - work_data.ADMITTIME
work_data['diff_time'] = work_data.diff_time.astype('timedelta64[s]').astype('int')
work_data = work_data.loc[work_data['diff_time']!=0,:].reset_index(drop=True)
work_data = work_data.drop(columns=['HADM_ID', 'ADMITTIME', 'outcome_time', 'death'])

compress_pickle.dump(work_data, 'Data/MIMIC_predictor_seq.gz')

""" output id of last visit for case-ctrl identification in R """
temp = work_data.loc[:,["SUBJECT_ID", "outcome_HADM"]].drop_duplicates()
temp.to_csv("Data/outcome_HADM.csv", index=False)

""" count data as predictor """
predictor = work_data.groupby('SUBJECT_ID').event.sum().reset_index()
uniq_subj = predictor.SUBJECT_ID
all_predictor = list(set([x for sublist in predictor.event for x in sublist]))

predictor_count = dict.fromkeys(all_predictor, 0)
predictor_count['SUBJECT_ID'] = 0
predictor_count = pd.DataFrame(predictor_count, index=[0])

# for all subject
for i, subj in enumerate(tqdm(uniq_subj)):
    temp = dict.fromkeys(all_predictor, 0)
    x = predictor.loc[predictor.SUBJECT_ID==subj,'event'].tolist()[0]
    temp_dict = {y_i: x.count(y_i) for y_i in set(x)}

    count_all = {k:temp.get(k, 0) + temp_dict.get(k, 0) for k in set(temp)}
    count_all['SUBJECT_ID'] = subj
    count_all = pd.DataFrame(count_all, index=[i+1])
    predictor_count = pd.concat([predictor_count, count_all])

# subject ID first column, remove the first row
cols = predictor_count.columns.tolist()
cols = cols[-1:] + cols[:-1]
predictor_count = predictor_count[cols]
predictor_count = predictor_count.iloc[1:]
predictor_count.index = predictor_count.index-1

compress_pickle.dump(predictor_count, 'Data/MIMIC_predictor_count.gz')


""" case-ctrl count from R """
cases = pyreadr.read_r("Data/outcome_cases.rds")[None]
cases["SUBJECT_ID"] = cases["SUBJECT_ID"].astype(int)
phecode_all = list(cases["phecode"].unique())

ctrls = pyreadr.read_r("Data/outcome_ctrls.rds")[None]
ctrls["SUBJECT_ID"] = ctrls["SUBJECT_ID"].astype(int)
phecode_all = set(phecode_all + list(ctrls["phecode"].unique()))

work_data = compress_pickle.load("Data/MIMIC_seq_data_unique.gz")
predictor_seq =  compress_pickle.load("Data/MIMIC_predictor_seq.gz")
predictor_seq = predictor_seq.groupby("SUBJECT_ID").event.agg(sum).reset_index()

pred_diag = work_data["event"].tolist()
pred_diag = set([x for sublist in pred_diag for x in sublist if "phecode" in x])
phecode_all = {x for x in phecode_all if "phecode_"+x in pred_diag}

outcome_lib = pd.read_csv("Data/phecode_info.csv", dtype=object)
outcome_lib = outcome_lib.drop(columns=["groupnum", "color"])
res = pd.DataFrame({"phecode":"xxx", "cases":0, "ctrls":0}, index=[0])


""" for all phecode, only keep incident cases """
for i, code in enumerate(tqdm(phecode_all)):
    tmp1 = cases.loc[cases["phecode"]==code,:]
    tmp2 = ctrls.loc[ctrls["phecode"]==code,:]
    
    need_to_remove = predictor_seq.loc[predictor_seq["event"].apply(lambda x: "phecode_"+code in x),:]
    tmp3 = tmp1.loc[tmp1["SUBJECT_ID"].apply(lambda x: x not in need_to_remove["SUBJECT_ID"]),:]
    tmp4 = tmp2.loc[tmp2["SUBJECT_ID"].apply(lambda x: x not in need_to_remove["SUBJECT_ID"]),:]
    
    tmp = pd.DataFrame(
        {"phecode":code, "cases":tmp3["SUBJECT_ID"].nunique(), "ctrls":tmp4["SUBJECT_ID"].nunique()},
        index=[i+1])
    res = pd.concat([res, tmp])

res = res.iloc[1:]
res.index = res.index-1

output = pd.merge(res, outcome_lib, how="left")
output["total"] = output["cases"]+output["ctrls"]
output["case_p"] = output["cases"]/output["total"]
output = output.sort_values(by=['case_p'], ascending=False)
output["phecode"] = ["phecode_"+x for x in output.phecode]
output = output.loc[:,["phecode", "cases", "ctrls", "total", "case_p", "description", "group"]]
output = output.loc[output["case_p"]>=0.0005]
output.to_csv("Data/MIMIC_outcome_count.csv", index=False)


""" case-ctrl status(outcome) from R """
cases = pyreadr.read_r("Data/outcome_cases.rds")[None]
cases["SUBJECT_ID"] = cases["SUBJECT_ID"].astype(int)
phecode_all = list(cases["phecode"].unique())

ctrls = pyreadr.read_r("Data/outcome_ctrls.rds")[None]
ctrls["SUBJECT_ID"] = ctrls["SUBJECT_ID"].astype(int)
phecode_all = set(phecode_all + list(ctrls["phecode"].unique()))

work_data = compress_pickle.load("Data/MIMIC_seq_data_unique.gz")
pred_diag = work_data["event"].tolist()
pred_diag = set([x for sublist in pred_diag for x in sublist if "phecode" in x])
phecode_all = {x for x in phecode_all if "phecode_"+x in pred_diag}


""" for all phecode, only keep incident cases """
res = pd.DataFrame(cases["SUBJECT_ID"].unique(), columns=["SUBJECT_ID"])
for i, code in enumerate(tqdm(phecode_all)):
    tmp1 = cases.loc[cases["phecode"]==code,:].drop(columns=["phecode"]).copy()
    tmp2 = ctrls.loc[ctrls["phecode"]==code,:].drop(columns=["phecode"]).copy()
    
    need_to_remove = predictor_seq.loc[predictor_seq["event"].apply(lambda x: "phecode_"+code in x),:]
    tmp3 = tmp1.loc[tmp1["SUBJECT_ID"].apply(lambda x: x not in need_to_remove["SUBJECT_ID"]),:].copy()
    tmp4 = tmp2.loc[tmp2["SUBJECT_ID"].apply(lambda x: x not in need_to_remove["SUBJECT_ID"]),:].copy()
    
    tmp3[code] = 1
    tmp4[code] = 0
    tmp = pd.concat([tmp3, tmp4]).reset_index(drop=True)
    res = pd.merge(res, tmp, how="left")

compress_pickle.dump(res, 'Data/MIMIC_outcome_case_ctrl.gz')