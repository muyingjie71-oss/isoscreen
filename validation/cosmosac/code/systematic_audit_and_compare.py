#!/usr/bin/env python3
"""Systematic frozen Layer A / NIST COSMO-SAC coverage audit and fair comparison.

Audit mode uses a fixed NIST first-block index intersection derived from the public
COSMOSAC/UD/complist.txt and the 747 molecule IDs in the freeze. It is an index
prefilter, NOT proof that a usable sigma3 profile or calculation exists.

Comparison mode ingests existing audited COSMO-SAC runs only and never fills
missing calculations. No training or changes to frozen OOF data.
"""
from __future__ import annotations
import csv, sys, re, json, glob, zipfile, io, math, os, contextlib, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
ARCHIVE=Path(os.environ.get('ISOMER_HANDOFF_ZIP', str(ROOT/'ORIGINAL_HANDOFF.zip')))


def load_snapshot():
 z=zipfile.ZipFile(ARCHIVE)
 def read(suffix):
  n=next(s for s in z.namelist() if s.endswith(suffix))
  return pd.read_csv(io.BytesIO(z.read(n)))
 return read('oof_observations_predictions.csv'),read('oof_pairs_predictions_33330.csv'),read('实验预测_跨溶剂27240对照.csv')


def get_partial_index():
    """First-block intersection of the frozen 747 IDs with NIST UD index.

    Metadata-derived first-block list, NOT an exact-key sigma3 inventory.
    """
    return set((ROOT/'input'/'NIST_firstblock_与冻结数据交集_356.txt').read_text(encoding='utf-8').split())


def add_flags(pairs, indexfirst):
 p=pairs.copy()
 fields=['Solute_Mol_ID_A','Solute_Mol_ID_B','Solvent_Mol_ID']
 for k in fields:
  p[k+'_prefix']=p[k].str[4:18]
  p[k+'_in_index']=p[k+'_prefix'].isin(indexfirst)
  p[k+'_std_neutral']=p[k].str.match(r'^MOL_[A-Z]{14}-UHFFFAOYSA-N$')
 p['NIST_firstblock_all3']=p[[k+'_in_index' for k in fields]].all(axis=1)
 p['NIST_neutral_ID_all3']=p[[k+'_std_neutral' for k in fields]].all(axis=1)
 p['NIST_candidate_neutral']=p.NIST_firstblock_all3 & p.NIST_neutral_ID_all3
 p['NIST_stage']='not_in_index_or_unsupported_identity'
 p.loc[p.NIST_firstblock_all3 & ~p.NIST_neutral_ID_all3,'NIST_stage']='identity_requires_special_handling'
 p.loc[p.NIST_candidate_neutral,'NIST_stage']='candidate_exact_full_key_and_sigma_pending'
 return p


def reuse_previous_cosmosac(pairs, obs):
 """Only prior *actual* calculations, joined by immutable Pair_ID or record IDs."""
 res=[];checks=[]
 # First C5/C6/C7 single point independent COSMO predictions
 sources=[
  str(ROOT/'input'/'C6_36.csv'),
  str(ROOT/'input'/'C7_27.csv'),
  str(ROOT/'input'/'C5_18.csv')
 ]
 single=pd.concat([pd.read_csv(s)[['canonical_record_id','ln_gamma_COSMOSAC']].assign(provenance=Path(s).name) for s in sources],ignore_index=True)
 duplicated=single.groupby('canonical_record_id').ln_gamma_COSMOSAC.agg(['max','min','count'])
 assert (duplicated['max']-duplicated['min']).abs().max()<1e-10
 single=single.drop_duplicates('canonical_record_id').set_index('canonical_record_id').ln_gamma_COSMOSAC
 pp=pairs[pairs.Canonical_Record_A.isin(single.index)&pairs.Canonical_Record_B.isin(single.index)].copy()
 pp['D_cosmo']=pp.Canonical_Record_A.map(single)-pp.Canonical_Record_B.map(single)
 pp['cosmo_provenance']='C5-C7 point models'
 res.append(pp[['Pair_ID','D_cosmo','cosmo_provenance']])
 checks.append({'input':'C5-C7 points','single_points':len(single),'derived_pair_matches':len(pp)})
 # Prior physical panel research (66 Pair_IDs) and independent source validation (124 Pair_IDs).
 panel=pd.read_csv(ROOT/'input'/'five_panels_66.csv')
 d=panel[['Pair_ID','D_COSMO_full']].dropna().rename(columns={'D_COSMO_full':'D_cosmo'}).copy()
 d['cosmo_provenance']='five physical panels'
 res.append(d[['Pair_ID','D_cosmo','cosmo_provenance']]);checks.append({'input':'five panels','input_pairs':len(d)})
 hold=pd.read_csv(ROOT/'input'/'seven_studies_124.csv')
 d=hold[['Pair_ID','D_COSMO_full','pair_orientation']].dropna().copy()
 d['D_cosmo']=d.D_COSMO_full*d.pair_orientation
 d['cosmo_provenance']='seven independent references'
 res.append(d[['Pair_ID','D_cosmo','cosmo_provenance']]);checks.append({'input':'seven studies','input_pairs':len(d)})
 result=pd.concat(res,ignore_index=True)
 result=result.merge(pairs[['Pair_ID','D_experimental','D_predicted']],how='left',on='Pair_ID',indicator=True,validate='many_to_one')
 assert (result['_merge']=='both').all(),'Cannot trace one or more COSMO predictions to frozen Pair_ID'
 dupes=result.groupby('Pair_ID').D_cosmo.agg(['max','min','count'])
 discrepancy=(dupes['max']-dupes['min']).abs()
 checks.append({'input':'cross-file repeats','n_repeated_ids':int((dupes['count']>1).sum()),'max_absolute_deviation':float(discrepancy.max())})
 # Abort rather than silently selecting a conflicting prediction.
 assert (discrepancy.max()<1e-5),f'COSMO values conflict across prior artifacts; max abs {discrepancy.max()}'
 result=result.sort_values('cosmo_provenance').drop_duplicates('Pair_ID')
 joined=pairs.merge(result[['Pair_ID','D_cosmo','cosmo_provenance']],on='Pair_ID',how='inner',validate='one_to_one')
 joined['ae_LayerA']=(joined.D_predicted-joined.D_experimental).abs()
 joined['ae_COSMO']=(joined.D_cosmo-joined.D_experimental).abs()
 joined['ae_Dzero']=joined.D_experimental.abs()
 joined['COSMO_better_than_LayerA']=joined.ae_COSMO<joined.ae_LayerA
 return joined,checks


def macro_table(df,groupname,groupcol):
 def group(g):
  return pd.Series({'n':len(g),'MAE_LayerA':g.ae_LayerA.mean(),'MAE_COSMO':g.ae_COSMO.mean(),'MAE_Dzero':g.ae_Dzero.mean(),'fraction_COSMO_better':g.COSMO_better_than_LayerA.mean()})
 o=df.groupby(groupcol,dropna=False).apply(group,include_groups=False).reset_index().rename(columns={groupcol:groupname})
 return o


def make_crosssolvent_edges(edges, pp):
 mapD=pp.set_index('Pair_ID')[['D_cosmo']].D_cosmo.to_dict()
 rx=re.compile(r'PAIRV2_[0-9a-f]+')
 # Some edges list multiple duplicate experiment rows. Keep only those where
 # all found/modelled pair IDs refer to one unambiguous matched D.
 valid=[]
 for r in edges.itertuples(index=False):
  a=rx.findall(str(r.original_Pair_IDs_S1));b=rx.findall(str(r.original_Pair_IDs_S2))
  if not a or not b or any(k not in mapD for k in a+b):continue
  v1=[mapD[k] for k in a];v2=[mapD[k] for k in b]
  if max(v1)-min(v1)>1e-7 or max(v2)-min(v2)>1e-7:continue
  valid.append([r.Edge_ID,r.Panel_ID,r.Family_ID,r.Isomer_Relation,r.Study_ID,r.T_K,r.Fold_ID,r.Solvent_1,r.Solvent_2,r.deltaD_exp_S2minusS1,r.deltaD_pred_S2minusS1,float(np.mean(v2)-np.mean(v1))])
 cols=['Edge_ID','Panel_ID','Family_ID','Isomer_Relation','Study_ID','T_K','Fold_ID','Solvent_1','Solvent_2','deltaD_exp','deltaD_LayerA','deltaD_COSMO']
 result=pd.DataFrame(valid,columns=cols)
 for lab,col in [('LayerA','deltaD_LayerA'),('COSMO','deltaD_COSMO'),('null','deltaD_exp')]:
  result['ae_'+lab]=(result[col]-result.deltaD_exp).abs() if lab!='null' else result.deltaD_exp.abs()
 return result


def main():
 obs,pairs,edges=load_snapshot()
 assert len(obs)==73243 and len(pairs)==33330 and len(edges)==27240
 assert obs.canonical_record_id.is_unique and pairs.Pair_ID.is_unique and edges.Edge_ID.is_unique
 index=get_partial_index();p=add_flags(pairs,index)
 neutral=p[p.NIST_candidate_neutral]
 cols=['Pair_ID','Family_ID','Fold_ID','Study_ID','Isomer_Relation','Isomer_A','Isomer_B','Solvent','T_K','D_experimental','D_predicted','Solute_Mol_ID_A','Solute_Mol_ID_B','Solvent_Mol_ID','Canonical_Record_A','Canonical_Record_B','Matched_Conditions','NIST_stage','NIST_firstblock_all3','NIST_neutral_ID_all3','NIST_candidate_neutral']
 p[cols].to_csv(ROOT/'01_全量33330对_NIST身份覆盖审计.csv',index=False)
 neutral[cols].to_csv(ROOT/'02_3681条候选匹配_待最终sigma确认.csv',index=False)
 summary=[]
 for key,g in [('全部冻结配对',p),('第一段存在索引',p[p.NIST_firstblock_all3]),('标准中性候选',neutral)]:
  summary.append({'cohort':key,'pairs':len(g),'families':g.Family_ID.nunique(),'studies':g.Study_ID.nunique(),'solvents':g.Solvent_Mol_ID.nunique(),'solute_molecules':len(set(g.Solute_Mol_ID_A)|set(g.Solute_Mol_ID_B)),'fraction_of_all_pairs':len(g)/len(p)})
 pd.DataFrame(summary).to_csv(ROOT/'03_覆盖汇总.csv',index=False)
 coverage_by_type=p.groupby('Isomer_Relation').agg(pairs_total=('Pair_ID','size'),pairs_candidate=('NIST_candidate_neutral','sum'),families=('Family_ID','nunique')).reset_index()
 coverage_by_type['candidate_fraction']=coverage_by_type.pairs_candidate/coverage_by_type.pairs_total
 coverage_by_type.to_csv(ROOT/'04_七类覆盖差异.csv',index=False)
 coverage_by_family=p.groupby(['Family_ID','Isomer_Relation'],dropna=False).agg(pairs_total=('Pair_ID','size'),pairs_candidate=('NIST_candidate_neutral','sum'),studies=('Study_ID','nunique')).reset_index()
 coverage_by_family['candidate_fraction']=coverage_by_family.pairs_candidate/coverage_by_family.pairs_total
 coverage_by_family.to_csv(ROOT/'05_各家族覆盖差异.csv',index=False)
 coverage_by_source=p.assign(Source=p.Source_A).groupby('Source').agg(pairs_total=('Pair_ID','size'),pairs_candidate=('NIST_candidate_neutral','sum')).reset_index()
 coverage_by_source.to_csv(ROOT/'06_数据来源覆盖.csv',index=False)
 # Available actual calculations in previous work; do NOT impute missing records.
 known,checks=reuse_previous_cosmosac(pairs,obs)
 known.to_csv(ROOT/'07_已计算COSMOSAC_严格PairID合并.csv',index=False)
 scoring=pd.DataFrame([{'group':'record_micro','n_groups':len(known),'MAE_LayerA':known.ae_LayerA.mean(),'MAE_COSMO':known.ae_COSMO.mean(),'MAE_Dzero':known.ae_Dzero.mean()},*[
  {'group':groupname+'_macro','n_groups':known[col].nunique(),'MAE_LayerA':known.groupby(col).ae_LayerA.mean().mean(),'MAE_COSMO':known.groupby(col).ae_COSMO.mean().mean(),'MAE_Dzero':known.groupby(col).ae_Dzero.mean().mean()}
  for groupname,col in [('family','Family_ID'),('study','Study_ID'),('type','Isomer_Relation')]]])
 scoring.to_csv(ROOT/'08_既有计算_总体同分母评分.csv',index=False)
 macro_table(known,'isomer_type','Isomer_Relation').to_csv(ROOT/'09_既有计算_七类性能.csv',index=False)
 macro_table(known,'family','Family_ID').to_csv(ROOT/'10_既有计算_家族性能.csv',index=False)
 macro_table(known,'study','Study_ID').to_csv(ROOT/'11_既有计算_逐文献性能.csv',index=False)
 cross=make_crosssolvent_edges(edges,known)
 cross.to_csv(ROOT/'12_既有计算_跨溶剂双差分边.csv',index=False)
 crosses=pd.DataFrame([{'group':'edge_micro','n_groups':len(cross),'MAE_LayerA':cross.ae_LayerA.mean() if len(cross) else None,'MAE_COSMO':cross.ae_COSMO.mean() if len(cross) else None,'MAE_zero_response':cross.ae_null.mean() if len(cross) else None},*[
 {'group':f'{x}_macro','n_groups':cross[k].nunique() if len(cross) else 0,'MAE_LayerA':cross.groupby(k).ae_LayerA.mean().mean() if len(cross) else None,'MAE_COSMO':cross.groupby(k).ae_COSMO.mean().mean() if len(cross) else None,'MAE_zero_response':cross.groupby(k).ae_null.mean().mean() if len(cross) else None}
 for x,k in [('family','Family_ID'),('study','Study_ID'),('panel','Panel_ID')]]])
 crosses.to_csv(ROOT/'13_既有计算_跨溶剂汇总.csv',index=False)
 qa={'input_file':str(ARCHIVE.name),'frozen_single_points':len(obs),'frozen_pairs':len(pairs),'frozen_crosssolvent_edges':len(edges),'published_NIST_firstblock_ids_in_freeze':len(index),'candidate_firstblock_pairs':int(p.NIST_firstblock_all3.sum()),'candidate_standard_neutral_pairs':int(p.NIST_candidate_neutral.sum()),'already_computed_pair_ids':len(known),'already_computed_edges':len(cross),'prior_artifact_merge_check':checks,'disclaimer':'Index coverage is not a completed sigma-profile calculation; model comparison only on actual previously computed pairs.'}
 (ROOT/'14_审计和计算核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(qa,ensure_ascii=False,indent=2))
 print(scoring.to_string(index=False))
 print(crosses.to_string(index=False))

if __name__=='__main__':main()
