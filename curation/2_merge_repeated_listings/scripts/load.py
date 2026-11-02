import pandas as pd, numpy as np, json
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from paths import AUDIT
COLS=['canonical_record_id','Study_ID','Original_DOI','Reference','Source_Original_Reference','Reported_Study_ID','Study_Identity_Basis','Source','source_list','Solute','Solvent','T_K','gamma_inf','ln_gamma_inf','Family_ID','Member_ID','Solute_Mol_ID','Solvent_Mol_ID','Solvent_Audit_ID','Solute_SMILES','Solvent_SMILES','Condition_Signature','Effective_Conditions','Duplicate_Class','Original_Measurement_Count','Raw_DOI','Raw_文献标题','Raw_期刊','Raw_发表年份','Raw_实验方法','Raw_标准态','Raw_活度系数不确定度','Raw_相态','Raw_压力_kPa','Raw_无限稀释判定方式','Study_Provenance_Status','QC_Status','QC_Reason','Fold_ID','Model_Eligible','Model_Eligibility_Reason','Original_Source_File','Original_Source_Location','Source03_Gamma_XML','Source03_T_XML']
def load_exp():
    d=pd.read_csv(AUDIT+'deduplicated_experiments.csv.gz',usecols=COLS,low_memory=False,dtype={'Study_ID':str},float_precision='round_trip')
    return d
def load_pairs():
    return pd.read_csv(AUDIT+'reviewed_model_task2.csv.gz',low_memory=False,float_precision='round_trip')
