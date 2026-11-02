#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Full-scale NIST/COSMO-SAC 2010/2014 on existing frozen isomer-pair OOF.

Usage with network:
 python run_full_cosmosac.py --source-archive ORIGINAL_HANDOFF.zip --network
Usage offline with populated NIST sigma cache:
 python run_full_cosmosac.py --source-archive ORIGINAL_HANDOFF.zip --cache PATH/TO/SIGMA_CACHE

Do NOT redistribute downloaded NIST profiles. Original NIST academic-use license
requires permission for redistribution; output archive excludes cache.
This script never trains or changes Layer A and never imputes missing COSMO data.
"""
from __future__ import annotations
import argparse, io, zipfile, re, pathlib, json, time, hashlib, os, sys, math, logging
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
HERE=pathlib.Path(__file__).resolve().parent
URL_BASE='https://raw.githubusercontent.com/usnistgov/COSMOSAC/master/profiles/UD/'
BIB='https://github.com/usnistgov/COSMOSAC'
SIGMA_DIR='sigma3'
DATA_DIR=HERE/'data'
DATA_FILES=('oof_observations_predictions.csv','oof_pairs_predictions_33330.csv','实验预测_跨溶剂27240对照.csv')


def read_zip(archive,suffix):
 with zipfile.ZipFile(archive) as z:
  n=next((x for x in z.namelist() if x.endswith('/'+suffix) or x==suffix),None)
  if n is None:raise KeyError(f'Missing {suffix} in {archive}')
  return pd.read_csv(io.BytesIO(z.read(n)))


def bundled_complete():
 return all((DATA_DIR/f).is_file() for f in DATA_FILES)


def source_mode(archive):
 # A complete self-contained copy of the three frozen data tables is shipped here.
 # If the user still pastes their old broken --source-archive path, fall back with notice.
 if archive is not None:
  arch=pathlib.Path(archive).expanduser()
  if arch.is_file():
   if not zipfile.is_zipfile(arch):
    raise RuntimeError(f'Input is not a valid ZIP: {arch}')
   print(f'[INPUT] Using existing original handoff ZIP: {arch.resolve()}',flush=True)
   return 'zip',arch
  if bundled_complete():
   print(f'[INPUT] Old handoff path not found: {arch}; using bundled frozen data instead.',flush=True)
   return 'bundled',DATA_DIR
  raise FileNotFoundError(f'Cannot find original ZIP {arch} and bundled data incomplete in {DATA_DIR}')
 if bundled_complete():
  print(f'[INPUT] Using bundled frozen data: {DATA_DIR}',flush=True)
  return 'bundled',DATA_DIR
 raise RuntimeError(f'Frozen data is missing from {DATA_DIR}. Extract the ENTIRE new zip, preserving data/ directory.')


def read_inputs(source):
 mode,location=source_mode(source)
 reader = (lambda name:read_zip(location,name)) if mode=='zip' else (lambda name:pd.read_csv(location/name))
 obs,pairs,edges=(reader(name) for name in DATA_FILES)
 if (len(obs),len(pairs),len(edges))!=(73243,33330,27240):
  raise RuntimeError(f'Frozen input row counts invalid: {[len(obs),len(pairs),len(edges)]}')
 if not obs.canonical_record_id.is_unique or not pairs.Pair_ID.is_unique or not edges.Edge_ID.is_unique:
  raise RuntimeError('Duplicate frozen identifiers found. Do not proceed with wrong inputs.')
 return obs,pairs,edges


def preflight(source,cache):
 obs,pairs,edges=read_inputs(source)
 print('[CHECK] Frozen inputs read successfully: observations=%d; pairs=%d; solvent edges=%d'%
       (len(obs),len(pairs),len(edges)),flush=True)
 print(f'[CHECK] Unique solute pair IDs={pairs.Pair_ID.nunique()}',flush=True)
 print(f'[CHECK] NIST cache path: {cache.resolve()}',flush=True)
 print('[CHECK] Cache index present:',(cache/'complist.txt').exists(),flush=True)
 print('[CHECK] No handoff.zip is required for this version.',flush=True)
 return True


def get_index(cache,network):
 cache.mkdir(parents=True,exist_ok=True)   # PATCH: the original script did not create this folder
 p=cache/'complist.txt'
 if not p.exists():
  if not network:
   raise RuntimeError('Official NIST index is not cached. Re-run with --network to download it.')
  print('[NIST] Downloading official compound index...',flush=True)
  try:
   raw_data=fetch_bytes(URL_BASE+'complist.txt')
  except Exception as exc:
   raise RuntimeError('Could not download official NIST index. Check your network access to raw.githubusercontent.com. Existing input data are fine.') from exc
  p.write_bytes(raw_data)
 raw=p.read_text(encoding='utf8')
 keys=set(re.findall(r'\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b',raw))
 if len(keys)<1000:
  raise RuntimeError(f'NIST index has suspiciously few compound IDs: {len(keys)}. Delete cache/complist.txt and retry.')
 print(f'[NIST] Index loaded: {len(keys)} unique full InChIKeys',flush=True)
 return keys,hashlib.sha256(raw.encode()).hexdigest()


def fetch_bytes(url,max_retries=4):
 last=None
 for i in range(max_retries):
  try:
   r=requests.get(url,timeout=(10,30),headers={'User-Agent':'Isomer-COSMOSAC-academic-comparison'});r.raise_for_status()
   return r.content
  except requests.exceptions.HTTPError as e:
   if e.response is not None and e.response.status_code in (400,401,403,404,410):
    raise RuntimeError(f'HTTP {e.response.status_code}: {url}') from e
   last=e
  except Exception as e:
   last=e
   if i<max_retries-1:time.sleep(1.5*(i+1))
 raise RuntimeError(f'Download failed {url}: {last}')


def parse_sigma_file(path,expected_key):
 text=path.read_text('utf8')
 lines=text.splitlines()
 assert lines[0].startswith('# meta: '),f'Bad metadata header in {path.name}'
 meta=json.loads(lines[0][len('# meta: '):])
 assert meta['standard_INCHIKEY']==expected_key, (expected_key,meta['standard_INCHIKEY'])
 data=[x.strip().split() for x in lines if x.strip() and not x.lstrip().startswith('#')]
 if len(data)!=153:raise ValueError(f'Expected 153 sigma bins in {path.name}, got {len(data)}')
 vals=np.asarray([[float(i) for i in x] for x in data],dtype=float)
 grid=np.tile(np.linspace(-.025,.025,51),3)
 assert np.max(np.abs(vals[:,0]-grid))<1e-8,'Wrong sigma grid'
 a=float(meta['area [A^2]']); v=float(meta['volume [A^3]'])
 if a<=0 or v<=0 or (vals[:,1]<-1e-10).any():raise ValueError('Invalid COSMO volume, area or distribution')
 if abs(vals[:,1].sum()/a-1)>3e-5:raise ValueError('Sigma profile area integral mismatch')
 nz=[(int(j),float(x)) for j,x in enumerate(vals[:,1]) if x>1e-13]
 return {'area':a,'volume':v,'dsp':float(meta.get('disp. e/kB [K]',0.0) or 0.0),
         'flag':meta.get('disp. flag','NHB'),'nz':nz,
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'full_inchikey':expected_key}


def fetch_one(cache,fullkey,network):
 p=cache/SIGMA_DIR/(fullkey+'.sigma');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  if not network:raise FileNotFoundError(p)
  p.write_bytes(fetch_bytes(URL_BASE+SIGMA_DIR+'/'+fullkey+'.sigma'))
 try:return parse_sigma_file(p,fullkey)
 except Exception:
  # remove potentially incomplete downloaded response; do not silently fix ID
  raise


def fetch_all(keys,cache,network,workers):
 avail={};errors={}
 if not keys:return avail,errors
 print(f'[NIST] Checking {len(keys)} sigma profiles (workers={workers}; network={network})...',flush=True)
 with ThreadPoolExecutor(max_workers=workers) as ex:
  futures={ex.submit(fetch_one,cache,k,network):k for k in sorted(keys)}
  for done,fut in enumerate(as_completed(futures),1):
   key=futures[fut]
   try:avail[key]=fut.result()
   except Exception as e:errors[key]=str(e)
   if done%25==0 or done==len(keys):
    print(f'[NIST] {done}/{len(keys)} checked; available={len(avail)} missing/failed={len(errors)}',flush=True)
 return avail,errors


def run_model(arch,out,cache,network,workers,limit=None):
 from cosmo3_core import SIGMAS,TYPES,R,comb_inf,profile,residual_inf,dispersion_inf,calc,PROFILES,validation
 # Use a separate module imported from this directory.
 import cosmo3_core as core
 obs,pairs,edge=read_inputs(arch)
 out.mkdir(parents=True,exist_ok=True)
 index,index_hash=get_index(cache,network)
 allowed=re.compile(r'^MOL_([A-Z]{14}-UHFFFAOYSA-N)$')
 def key_of(s):
  m=allowed.fullmatch(str(s));return m.group(1) if m else None
 for f in ('Solute_Mol_ID_A','Solute_Mol_ID_B','Solvent_Mol_ID'):
  pairs[f+'_fullkey']=pairs[f].map(key_of)
  pairs[f+'_index_covered']=pairs[f+'_fullkey'].isin(index)
 eligible=pairs[pairs[[f+'_index_covered' for f in ('Solute_Mol_ID_A','Solute_Mol_ID_B','Solvent_Mol_ID')]].all(axis=1)].copy()
 if limit is not None:
  # Optional debug subset, deterministic ID ordering; never use for reported full results.
  eligible=eligible.sort_values('Pair_ID').head(limit).copy()
 keys=set(eligible.Solute_Mol_ID_A_fullkey)|set(eligible.Solute_Mol_ID_B_fullkey)|set(eligible.Solvent_Mol_ID_fullkey)
 if not len(eligible):
  raise RuntimeError('No valid NIST candidate pairs in frozen data. Check that the NIST index and the correct source inputs are being used.')
 print(f'NIST strict identity candidate pairs={len(eligible)}, unique molecules={len(keys)}',flush=True)
 profiles,failed_profiles=fetch_all(keys,cache,network,workers)
 core.PROFILES.clear()
 core.PROFILES.update(profiles)
 # dispersion compatibility with NIST implementation: the default core explicitly
 # excluded water and COOH; correct for the full 46-family analysis here.
 def dispersion_full(name_s,name_v):
  u=core.PROFILES[name_s];v=core.PROFILES[name_v]
  c0,c1=u['flag'],v['flag'];w=0.27027
  if ('H2O' in (c0,c1) and ('HB-ACCEPTOR' in (c0,c1) or 'COOH' in (c0,c1))) or ('COOH' in (c0,c1) and ('NHB' in (c0,c1) or 'HB-DONOR-ACCEPTOR' in (c0,c1))):w=-w
  es=u['dsp'];ev=v['dsp']
  return w*(0.5*(es+ev)-math.sqrt(max(0,es*ev)))
 @lru_cache(maxsize=120000)
 def model_one(solvent,solute,T):
  if solute==solvent:return 0.0,0.0,0.0,0.0
  u,v=core.PROFILES[solute],core.PROFILES[solvent]
  C=core.comb_inf(u,v)
  R_val,n_i,n_s=core.residual_inf(T,solute,solvent)
  dsp=dispersion_full(solute,solvent)
  z=C+R_val+dsp
  if not np.isfinite(z):raise ValueError('Nonfinite ln gamma')
  return float(z),float(C),float(R_val),float(dsp)
 qa={'official_index_sha256':index_hash,'NIST_index_fullkeys':len(index),
     'candidate_pairs':len(eligible),'mol_key_count':len(keys),'profiles_available':len(profiles),
     'profile_failures':len(failed_profiles),'frozen_OOF_untouched':True,
     'NIST_repo':BIB,'model':'NIST COSMO3 2010 Hsieh + dispersion correction',
     'status':'pending'}
 # Reference benchmark must pass even when some requested profiles are unavailable.
 try:
  bench_profiles,be=fetch_all(['OFBQJSOFQDEBGM-UHFFFAOYSA-N','IMNFDUFMRHMDMM-UHFFFAOYSA-N'],cache,network,workers)
  core.PROFILES.update(bench_profiles)
  if be:raise RuntimeError('Official numerical reference profile unavailable: '+str(be))
  z=model_one('IMNFDUFMRHMDMM-UHFFFAOYSA-N','OFBQJSOFQDEBGM-UHFFFAOYSA-N',293.15)
  if max(abs(z[1]+0.0317),abs(z[2]-0.0001),abs(z[3]))>=0.002:
   raise RuntimeError(f'NIST reference test failed, numerical result {z}')
  qa['NIST_reference_293.15K']=z
  print('[CHECK] NIST reference numerical benchmark PASSED',flush=True)
 except Exception as e:
  qa['reference_error']=str(e)
  (out/'qa_profile_availability.json').write_text(json.dumps({'metadata':qa,'failed_profiles':failed_profiles},ensure_ascii=False,indent=2),encoding='utf8')
  raise RuntimeError('Cannot start validated production predictions: '+str(e)) from e
 (out/'qa_profile_availability.json').write_text(json.dumps({'metadata':qa,'failed_profiles':failed_profiles},ensure_ascii=False,indent=2),encoding='utf8')
 rows=[];failures=[]
 unique=eligible[['Canonical_Record_A','Canonical_Record_B','Solute_Mol_ID_A_fullkey','Solute_Mol_ID_B_fullkey','Solvent_Mol_ID_fullkey','T_K']].copy()
 for i,row in enumerate(eligible.itertuples(index=False)):
  a=row.Solute_Mol_ID_A_fullkey;b=row.Solute_Mol_ID_B_fullkey;s=row.Solvent_Mol_ID_fullkey;T=float(row.T_K)
  if any(k in failed_profiles for k in (a,b,s)):
   failures.append({'Pair_ID':row.Pair_ID,'reason':'sigma_file_unavailable_or_identity_failed'});continue
  try:
   A=model_one(s,a,T);B=model_one(s,b,T)
   D=A[0]-B[0]
   rows.append({'Pair_ID':row.Pair_ID,'D_cosmo':D,'D_cosmo_comb':A[1]-B[1],
       'D_cosmo_resid':A[2]-B[2],'D_cosmo_disp':A[3]-B[3],
       'gammaA_COSMO':A[0],'gammaB_COSMO':B[0]})
  except Exception as e:
   failures.append({'Pair_ID':row.Pair_ID,'reason':repr(e)[:240]})
  if (i+1)%200==0:
   print(f'Computed {i+1}/{len(eligible)}; successful={len(rows)}, failed={len(failures)}',flush=True)
   pd.DataFrame(rows).to_csv(out/'checkpoint_partial_results.csv',index=False)
 if not rows:
  qa['status']='no_calculations_possible';qa['failures']=len(failures)
  (out/'qa_final.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
  raise RuntimeError('No COSMO-SAC predictions; inspect sigma availability log')
 calcdf=pd.DataFrame(rows)
 final=eligible.merge(calcdf,on='Pair_ID',how='inner',validate='one_to_one')
 final['ae_COSMO']=(final.D_experimental-final.D_cosmo).abs()
 final['ae_LayerA']=(final.D_experimental-final.D_predicted).abs()
 final['ae_Dzero']=final.D_experimental.abs()
 final.to_csv(out/'COSMO_LayerA_pairs_matched.csv',index=False)
 pd.DataFrame(failures, columns=['Pair_ID','reason']).to_csv(out/'failed_pairs.csv',index=False)
 # All score values computed on identical Pair_IDs, micro and group macro.
 s=[]
 for label,key in [('record_micro',None),('family_macro','Family_ID'),('source_macro','Study_ID'),('type_macro','Isomer_Relation')]:
  v=final[['ae_COSMO','ae_LayerA','ae_Dzero']].mean() if key is None else final.groupby(key)[['ae_COSMO','ae_LayerA','ae_Dzero']].mean().mean()
  s.append({'cohort':label,'groups':len(final) if key is None else final[key].nunique(),**v.to_dict()})
 pd.DataFrame(s).to_csv(out/'summary_scores.csv',index=False)
 # No extrapolation or aggregation of pair/solvent differences across papers.
 from systematic_audit_and_compare import make_crosssolvent_edges
 cross=make_crosssolvent_edges(edge,final)
 cross.to_csv(out/'COSMO_LayerA_crosssolvent_edges.csv',index=False)
 qa['status']='complete_for_available_profiles';qa.update({'computed_pairs':len(final),'failed_pairs':len(failures),'matched_crosssolvent_edges':len(cross)})
 (out/'qa_final.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(qa,ensure_ascii=False,indent=2),flush=True)


def main():
 ap=argparse.ArgumentParser(description='Self-contained frozen Layer A vs NIST COSMO-SAC comparison')
 ap.add_argument('--source-archive',type=pathlib.Path,default=None,
                 help='Optional original handoff zip. Not needed: frozen CSVs are bundled in data/.')
 ap.add_argument('--cache',type=pathlib.Path,default=HERE/'private_sigma_cache')
 ap.add_argument('--out',type=pathlib.Path,default=HERE/'bulk_calculation_outputs')
 ap.add_argument('--network',action='store_true')
 ap.add_argument('--workers',type=int,default=4)
 ap.add_argument('--check-only',action='store_true',help='Verify input files; do not download or compute.')
 ap.add_argument('--limit',type=int,default=None,help='SMOKE TEST ONLY: never report subset as final results')
 a=ap.parse_args()
 if a.workers<1:ap.error('--workers must be >=1')
 if a.limit is not None and a.limit<1:ap.error('--limit must be >=1')
 if a.check_only:
  preflight(a.source_archive,a.cache)
  return
 print('[START] Running frozen Layer A / NIST COSMO-SAC; no re-training and no quantum chemistry.',flush=True)
 run_model(a.source_archive,a.out,a.cache,a.network,a.workers,a.limit)
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  print('\n[ERROR] '+str(e),file=sys.stderr,flush=True)
  print('[HELP] For package issues, run: py run_full_cosmosac.py --check-only',file=sys.stderr,flush=True)
  print('[HELP] NIST downloads require Internet access to raw.githubusercontent.com.',file=sys.stderr,flush=True)
  traceback.print_exc()
  try:
   log=HERE/'last_error.txt'
   log.write_text('COSMO-SAC calculation failed\n\n'+traceback.format_exc(),encoding='utf-8')
   print(f'[ERROR-LOG] Full traceback saved to: {log}',file=sys.stderr,flush=True)
  except Exception:
   pass
  sys.exit(1)
