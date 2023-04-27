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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from datetime import datetime

import pyreadr

""" output diagnosis ICD for phenotype mapping in R """
# diag = pd.read_csv("../../Data/MIMIC_III/DIAGNOSES_ICD.csv.gz")
# diag = diag.iloc[:,[1,2,4]]

# phe_map = pd.read_csv("Data/phecode_ICD9.csv", dtype=object).drop(columns="vocabulary_id")
# phe_map["ICD9_CODE"] = phe_map["code"].apply(lambda x: x.replace(".", ""))
# diag = pd.merge(diag, phe_map, how='left')

# diag = diag.drop(columns=["ICD9_CODE", "phecode"]).rename(columns={"code":"ICD9"})
# diag = diag.dropna().drop_duplicates()

# sex = pd.read_csv("../../Data/MIMIC_III/PATIENTS.csv.gz")
# sex = sex.loc[:,['SUBJECT_ID', 'GENDER']]
# diag = pd.merge(diag, sex, how='left').drop(columns='SUBJECT_ID')

# diag.to_csv("Data/all_diag_ICD9.csv", index=False)
# diag.HADM_ID.nunique()

""" adms """
adms = pd.read_csv("../../Data/MIMIC_III/ADMISSIONS.csv.gz")
adms = adms.iloc[:,[1,2,3]].drop_duplicates()

""" phenotype """
diag = pyreadr.read_r("Data/all_diag_phecode.rds")[None]
diag["HADM_ID"] = diag["HADM_ID"].astype(int)
diag = diag.dropna().drop_duplicates()
diag = pd.merge(diag, adms.loc[:,["SUBJECT_ID", "HADM_ID"]], how="left")

diag_new = diag.groupby(['SUBJECT_ID', 'HADM_ID']).apply(lambda x: list(x.phecode)).reset_index()
diag_new = diag_new.rename(columns={0:'phecode'})
diag_new = pd.merge(diag_new, adms, on=['SUBJECT_ID', 'HADM_ID'])

""" pres """
pres = pd.read_csv("../../Data/MIMIC_III/PRESCRIPTIONS.csv.gz")
pres = pres.iloc[:,[1,2,10]]
pres = pres.dropna().drop_duplicates()

pres_new = pres.groupby(['SUBJECT_ID', 'HADM_ID']).apply(lambda x: list(x.FORMULARY_DRUG_CD)).reset_index()
pres_new = pres_new.rename(columns={0: 'DRUG'})
pres_new = pd.merge(pres_new, adms, on=['SUBJECT_ID', 'HADM_ID'])

""" labs """
labs = pd.read_csv("../../Data/MIMIC_III/LABEVENTS.csv.gz")
labs = labs.iloc[:,[1,2,3]]
labs = labs.dropna().drop_duplicates()

labs_new = labs.groupby(["SUBJECT_ID", "HADM_ID"]).apply(lambda x: list(x.ITEMID)).reset_index()
labs_new = labs_new.rename(columns={0: 'ITEMID'})
labs_new = pd.merge(labs_new, adms, on=['SUBJECT_ID', 'HADM_ID'])

compress_pickle.dump([diag_new, pres_new, labs_new], "Data/MIMIC_list_data.gz")


""" preprocess the data, unique concept for each visit """
diag, pres, labs = compress_pickle.load("Data/MIMIC_list_data.gz")
adms = pd.read_csv("../../Data/MIMIC_III/ADMISSIONS.csv.gz")
adms = adms.iloc[:,[1,2,3]]

diag.phecode = diag.phecode.apply(lambda x: list(set(x)))
pres.DRUG = pres.DRUG.apply(lambda x: list(set(x)))
labs.ITEMID = labs.ITEMID.apply(lambda x: list(set(x)))

uniq_diag = diag.phecode.tolist()
uniq_diag = [x for sublist in uniq_diag for x in sublist]
count_diag = pd.Series(uniq_diag).value_counts()

uniq_pres = pres.DRUG.tolist()
uniq_pres = [x for sublist in uniq_pres for x in sublist]
count_pres = pd.Series(uniq_pres).value_counts()

uniq_labs = labs.ITEMID.tolist()
uniq_labs = [x for sublist in uniq_labs for x in sublist]
count_labs = pd.Series(uniq_labs).value_counts()

# filter freq
n = 50
count_diag_filter = count_diag[count_diag>=n]
count_pres_filter = count_pres[count_pres>=n]
count_labs_filter = count_labs[count_labs>=n]

uniq_diag = list(count_diag_filter.index)
uniq_pres = list(count_pres_filter.index)
uniq_labs = list(count_labs_filter.index)

diag.phecode = diag.phecode.apply(lambda x: [code for code in x if code in uniq_diag])
pres.DRUG = pres.DRUG.apply(lambda x: [code for code in x if code in uniq_pres])
labs.ITEMID = labs.ITEMID.apply(lambda x: [code for code in x if code in uniq_labs])

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
for i, ax in zip([diag.phecode, pres.DRUG, labs.ITEMID], axes.flatten()):
    plot_data = pd.DataFrame({"num": i.apply(lambda x:len(x))})
    sns.histplot(data=plot_data, x="num", ax=ax)
plt.show()

""" combine three """
work_data = pd.merge(adms, diag.iloc[:,[0,1,2]], how='left')
work_data = pd.merge(work_data, pres.iloc[:,[0,1,2]], how='left')
work_data = pd.merge(work_data, labs.iloc[:,[0,1,2]], how='left')

work_data.phecode = work_data.phecode.fillna('0').apply(lambda x:['phecode_'+str(code) for code in x])
work_data.DRUG = work_data.DRUG.fillna('0').apply(lambda x:['DRUG_'+str(code) for code in x])
work_data.ITEMID = work_data.ITEMID.fillna('0').apply(lambda x:['LAB_'+str(code) for code in x])

work_data['event'] = work_data.phecode+work_data.DRUG+work_data.ITEMID
work_data['event'] = work_data.event.apply(
    lambda x: [code for code in x if (code not in ['phecode_0','DRUG_0','LAB_0'])]
    )
idx = work_data.event.apply(lambda x:len(x)==0)
work_data = work_data.loc[~idx]
work_data.ADMITTIME = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in list(work_data.ADMITTIME)]

""" save sequence data """
work_data_new = work_data[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'event']]
compress_pickle.dump(work_data_new, 'Data/MIMIC_seq_data_unique.gz')