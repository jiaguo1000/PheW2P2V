#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: jiaguo """

# import Modules
import os
import math
import timeit
import compress_pickle
import itertools
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from time import sleep

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from scipy.special import expit
from scipy.special import comb

from PheW2P2V import *
import xgboost as xgb

def get_metric(y_true, y_score, note):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_all = 2*recall*precision/(recall+precision)
    f1 = np.nanmax(f1_all)
    p = precision[np.nanargmax(f1_all)]
    r = recall[np.nanargmax(f1_all)]
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    res = {note+"_f1": f1, note+"_p": p, note+"_r": r, note+"_ap": ap, note+"_auc": auc}
    return res

def cv_each_D(
    cv_i, case_ID, ctrl_ID, case_tran_test, ctrl_tran_test, count_data, seq_data, seq_data_all
    ):
    """ tran test ID """
    case_tran_ID = [case_ID[i] for i in case_tran_test[cv_i][0]]
    case_test_ID = [case_ID[i] for i in case_tran_test[cv_i][1]]

    ctrl_tran_ID = [ctrl_ID[i] for i in ctrl_tran_test[cv_i][0]]
    ctrl_test_ID = [ctrl_ID[i] for i in ctrl_tran_test[cv_i][1]]

    """ LR and RF """
    X_tran = pd.DataFrame({"SUBJECT_ID":case_tran_ID+ctrl_tran_ID})
    X_tran = pd.merge(X_tran, count_data, how="left").drop(columns=["SUBJECT_ID"])
    X_test = pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID})
    X_test = pd.merge(X_test, count_data, how="left").drop(columns=["SUBJECT_ID"])

    y_tran = [1]*len(case_tran_ID) + [0]*len(ctrl_tran_ID)
    y_test = [1]*len(case_test_ID) + [0]*len(ctrl_test_ID)

    LR_model = LogisticRegression(random_state=0, penalty="l1", solver="liblinear").fit(X_tran, y_tran)
    LR = get_metric(y_test, LR_model.predict_proba(X_test)[:,1], "LR")
    
    RF_model = RandomForestClassifier(random_state=0).fit(X_tran, y_tran)
    RF = get_metric(y_test, RF_model.predict_proba(X_test)[:,1], "RF")

    XGB_model = xgb.XGBClassifier(random_state=0).fit(X_tran, y_tran)
    XGB = get_metric(y_test, XGB_model.predict_proba(X_test)[:,1], "XGB")

    """ seq prediction """
    """ train word2vec """
    seq_test = pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID})
    seq_test = pd.merge(seq_test, seq_data, how="left")
    seq_test = seq_test.drop(columns=["time_rank", "outcome_HADM"])
    train_lib = seq_data_all.loc[~seq_data_all["SUBJECT_ID"].isin(seq_test["SUBJECT_ID"]),]
    train_lib = train_lib.seq.tolist()
    
    test_PV = PheW2P2V(train_lib=train_lib, seq_test=seq_test, disease=disease, win_size=500, vec_size=200, n_epoch=1)
    assert sum(test_PV.SUBJECT_ID==pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID}).SUBJECT_ID)==test_PV.shape[0]
    
    P2V = get_metric(y_test, test_PV["PV_no_time_no_corr"], "P2V")
    WP2V = get_metric(y_test, test_PV["PV_no_time_with_corr"], "WP2V")
    
    """ res AUC """
    res = {}
    res.update(LR); res.update(RF); res.update(XGB); res.update(P2V); res.update(WP2V)
    return(res)

task_id = int(os.getenv("SGE_TASK_ID"))
np.random.seed(1)
random.seed(1)

""" read data """
seq_data_all = compress_pickle.load("Data/MIMIC_seq_data_unique.gz")
seq_data_all["event"] = seq_data_all["event"].apply(lambda x: random.sample(x, k=len(x)))
max(seq_data_all["event"].apply(lambda x:len(x)))
seq_data_all = seq_data_all.groupby("SUBJECT_ID").event.agg(sum).reset_index().rename(columns={"event":"seq"})

seq_data = compress_pickle.load("Data/MIMIC_predictor_seq.gz")
count_data = compress_pickle.load("Data/MIMIC_predictor_count.gz")

outcome_data = compress_pickle.load("Data/MIMIC_outcome_case_ctrl.gz")
outcome_list = pd.read_csv("Data/MIMIC_outcome_count.csv")
disease_list = outcome_list['phecode'].tolist()
len(disease_list)

""" each disease """
i = task_id-1
disease = disease_list[i]
case_ID = outcome_data.loc[outcome_data[disease[8:]]==1, "SUBJECT_ID"].tolist()
ctrl_ID = outcome_data.loc[outcome_data[disease[8:]]==0, "SUBJECT_ID"].tolist()
n_case = len(case_ID)
n_ctrl = len(ctrl_ID)
assert n_case+n_ctrl==outcome_list.loc[outcome_list["phecode"]==disease, "total"].values, "Error!"
print("{} | disease = {} | num_case = {} | num_ctrl = {}".format(i, disease, n_case, n_ctrl), flush=True)

""" split 10 fold """
i_rep = 10
case_tran_test = []
ctrl_tran_test = []
for seed in range(i_rep):
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    case_tran_test.append(list(kf.split(case_ID))[0])
    ctrl_tran_test.append(list(kf.split(ctrl_ID))[0])

""" combine """
for cv_i in range(i_rep):
    prll_res = cv_each_D(
        cv_i=cv_i, case_ID=case_ID, ctrl_ID=ctrl_ID,
        case_tran_test=case_tran_test, ctrl_tran_test=ctrl_tran_test,
        count_data=count_data, seq_data=seq_data, seq_data_all=seq_data_all
    )
    temp = pd.DataFrame(prll_res, index=[cv_i])
    res_AUC = temp if cv_i==0 else pd.concat([res_AUC, temp])

output = res_AUC.copy()
output["phecode"] = disease

cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output["rank"] = task_id

""" output """
filename = "../../Output/MIMIC_task_{}.csv".format(
    task_id
)

output.to_csv(filename, index=False)