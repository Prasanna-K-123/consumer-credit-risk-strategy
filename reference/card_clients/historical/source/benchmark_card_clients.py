"""Equal-budget, duplicate-group-separated credit model comparison.

One 2005 Taiwan cohort; not out-of-time validation. Financial decision costs
are illustrative error weights, not currency or realised lending returns.
"""
import argparse
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from src.woe import WOETransformer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reference/card_clients'
EXCLUDED={'ID','SEX','EDUCATION','MARRIAGE'}
TARGET='default payment next month'
SEED=20261010


def partition(groups):
    groups=np.asarray(groups)
    ids=np.arange(len(groups))
    fit,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=SEED).split(ids,groups=groups))
    rest_dev,rest_test=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+1).split(rest,groups=groups[rest]))
    dev,test=rest[rest_dev],rest[rest_test]
    cal_local,policy_local=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+2).split(dev,groups=groups[dev]))
    result=dict(train=fit,calibration=dev[cal_local],policy_validation=dev[policy_local],test=test)
    assert sum(len(x) for x in result.values())==len(groups)
    names=list(result)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not set(groups[result[a]])&set(groups[result[b]]),'Cross-partition feature-group leakage'
    return result


def logits(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    if not np.isfinite(p).all():raise ValueError('Nonfinite probabilities')
    return np.log(p/(1-p)).reshape(-1,1)


def fit_calibration(p,y):
    if len(np.unique(y))!=2:raise ValueError('Calibration needs both outcomes')
    return LogisticRegression(C=1.,max_iter=2000,random_state=SEED).fit(logits(p),y)


def cost(y,p,threshold):
    y=np.asarray(y,dtype=int); reject=np.asarray(p)>=threshold
    return 5*((y==1)&~reject).astype(int)+((y==0)&reject).astype(int)


def select_threshold(y,p):
    grid=np.arange(1,100)/100
    totals=[int(cost(y,p,t).sum()) for t in grid]
    i=int(np.argmin(totals))
    return float(grid[i]),int(totals[i])


def metrics(y,p,threshold):
    y=np.asarray(y,dtype=int);reject=np.asarray(p)>=threshold
    return dict(auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)),
        threshold=float(threshold),false_negative=int(((y==1)&~reject).sum()),false_positive=int(((y==0)&reject).sum()),
        accepted=int((~reject).sum()),cost_units=int(cost(y,p,threshold).sum()))


def paired_group_bootstrap(y,a,b,ta,tb,groups,n=2000):
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(SEED)
    scores=[]
    for _ in range(n):
        idx=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
        yi=y[idx]
        if len(np.unique(yi))!=2:raise ValueError('One-class bootstrap draw; audit required')
        scores.append([roc_auc_score(yi,b[idx])-roc_auc_score(yi,a[idx]),
                       np.mean((b[idx]-yi)**2-(a[idx]-yi)**2),
                       float(np.mean(cost(yi,b[idx],tb)-cost(yi,a[idx],ta)))])
    point=[float(roc_auc_score(y,b)-roc_auc_score(y,a)),float(np.mean((b-y)**2-(a-y)**2)),float(np.mean(cost(y,b,tb)-cost(y,a,ta)))]
    interval=np.quantile(np.asarray(scores),[.025,.975],axis=0)
    return {name:dict(challenger_minus_scorecard=point[i],ci95=interval[:,i].tolist()) for i,name in enumerate(['auc','brier','cost_per_client'])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    data=ROOT/'data/uci_card_clients.xls'
    if a.download:
        raw=urllib.request.urlopen('https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',timeout=60).read()
        import io
        z=zipfile.ZipFile(io.BytesIO(raw)); raw=z.read([n for n in z.namelist() if n.endswith('.xls')][0])
        data.parent.mkdir(parents=True,exist_ok=True);data.write_bytes(raw)
    assert hashlib.sha256(data.read_bytes()).hexdigest()==protocol['xls_sha256']
    d=pd.read_excel(data,header=1)
    assert d.shape==(30000,25) and not d.isna().any().any() and d['ID'].is_unique
    y=d[TARGET].to_numpy(dtype=int);assert set(y)=={0,1}
    features=[x for x in d.columns if x not in EXCLUDED|{TARGET}]
    assert len(features)==20
    X=d[features].copy()
    groups=pd.util.hash_pandas_object(X,index=False).astype(str).to_numpy()
    parts=partition(groups)
    categorical=[c for c in features if c.startswith('PAY_') and not c.startswith('PAY_AMT')]
    continuous=[c for c in features if c not in categorical]
    fit,cal,val,test=[parts[k] for k in ['train','calibration','policy_validation','test']]
    started=time.perf_counter()
    transform=WOETransformer(continuous_features=continuous,categorical_features=categorical,n_bins=5)
    Z=transform.fit_transform(X.iloc[fit],pd.Series(y[fit],index=X.iloc[fit].index))
    scorecard=LogisticRegression(C=1.,max_iter=3000,random_state=SEED).fit(Z,y[fit])
    challenger=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.05,l2_regularization=10.,early_stopping=False,random_state=SEED).fit(X.iloc[fit],y[fit])
    models={'scorecard':lambda data:scorecard.predict_proba(transform.transform(data))[:,1],
            'challenger':lambda data:challenger.predict_proba(data)[:,1]}
    preds={};thresholds={};calibration_info={};results={};cal_bins=[]
    for name,predict in models.items():
        rawcal=predict(X.iloc[cal]);calibrator=fit_calibration(rawcal,y[cal])
        pval=calibrator.predict_proba(logits(predict(X.iloc[val])))[:,1]
        threshold,valcost=select_threshold(y[val],pval)
        rawtest=predict(X.iloc[test]);p=calibrator.predict_proba(logits(rawtest))[:,1]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        preds[name]=p;thresholds[name]=threshold
        results[name]=metrics(y[test],p,threshold)
        results[name]['raw_test_brier']=float(brier_score_loss(y[test],rawtest))
        calibration_info[name]=dict(intercept=float(calibrator.intercept_[0]),slope=float(calibrator.coef_[0,0]),validation_cost=valcost)
        for i in range(10):
            selected=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
            n=int(selected.sum())
            cal_bins.append(dict(model=name,bin_left=i/10,bin_right=(i+1)/10,n=n,mean_probability=float(p[selected].mean()) if n else None,default_rate=float(y[test][selected].mean()) if n else None))
    raw_predictions=pd.DataFrame({'source_id':d.ID.iloc[test].to_numpy(),'feature_group':groups[test],'default':y[test],'SEX_for_audit_only':d.SEX.iloc[test].to_numpy(),'AGE':d.AGE.iloc[test].to_numpy(),**preds})
    comparisons=paired_group_bootstrap(y[test],preds['scorecard'],preds['challenger'],thresholds['scorecard'],thresholds['challenger'],groups[test])
    group_audit=[]
    for label,mask in [('sex_1',d.SEX.iloc[test].to_numpy()==1),('sex_2',d.SEX.iloc[test].to_numpy()==2),('age_under30',d.AGE.iloc[test].to_numpy()<30),('age_30to49',(d.AGE.iloc[test].to_numpy()>=30)&(d.AGE.iloc[test].to_numpy()<50)),('age_50plus',d.AGE.iloc[test].to_numpy()>=50)]:
        for name,p in preds.items():
            m=metrics(y[test][mask],p[mask],thresholds[name]);m.update(model=name,group=label,n=int(mask.sum()),default_rate=float(y[test][mask].mean()),false_negative_rate=m['false_negative']/int(y[test][mask].sum()),false_positive_rate=m['false_positive']/int((1-y[test][mask]).sum()))
            group_audit.append(m)
    split=pd.DataFrame({'source_id':d.ID,'feature_group':groups,'partition':''})
    for name,idx in parts.items():split.loc[idx,'partition']=name
    csvs={'partition_manifest.csv':split.to_csv(index=False),'holdout_predictions.csv':raw_predictions.to_csv(index=False,float_format='%.12g'),'calibration_bins.csv':pd.DataFrame(cal_bins).to_csv(index=False),'group_audit.csv':pd.DataFrame(group_audit).to_csv(index=False)}
    summary=dict(source=protocol['source'],data_sha256=protocol['xls_sha256'],license='CC BY 4.0; I-Cheng Yeh, UCI DOI10.24432/C55S3H',
        rows=len(d),features=features,excluded=sorted(EXCLUDED),unique_model_input_groups=len(np.unique(groups)),duplicate_input_rows=len(d)-len(np.unique(groups)),
        partitions={k:dict(rows=len(v),groups=len(np.unique(groups[v])),defaults=int(y[v].sum()),default_rate=float(y[v].mean())) for k,v in parts.items()},
        cross_partition_duplicate_groups=0,models=results,calibration=calibration_info,paired_group_bootstrap_replicates=2000,paired_uncertainty=comparisons,
        constant_policies=dict(accept_all_cost_units=int(y[test].sum()*5),reject_all_cost_units=int((1-y[test]).sum())),
        protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest(),
        file_sha256={name:hashlib.sha256(raw.encode()).hexdigest() for name,raw in csvs.items()},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        scope=protocol['scope'],uncertainty_scope='Fixed fitted models and calibration/thresholds; group resampling does not include training/refit uncertainty.',
        fair_comparison='Both models use identical 20 predictors and exactly the same training/calibration/policy-validation/test clients. No test-driven tuning. Separate preprocessing is part of each prespecified model.')
    if a.verify:
        frozen=json.loads((OUT/'summary.json').read_text())
        # Fit code can vary by platform; require exact split/decision counts and
        # matched-budget metrics to 1e-7, not identical host/elapsed metadata.
        for key in ['partitions','rows','unique_model_input_groups','duplicate_input_rows','constant_policies','cross_partition_duplicate_groups','features','excluded']:
            assert summary[key]==frozen[key],key
        for name in models:
            for key,value in results[name].items():
                assert np.isclose(value,frozen['models'][name][key],rtol=0,atol=1e-7),(name,key,value,frozen['models'][name][key])
        assert csvs['partition_manifest.csv']==(OUT/'partition_manifest.csv').read_text()
        print('MATCHED_CREDIT_REPRODUCTION_PASS')
    else:
        for name,text in csvs.items():(OUT/name).write_text(text)
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (OUT/'runtime.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started,'cpu_thread_budget':'OMP_NUM_THREADS=2; OPENBLAS_NUM_THREADS=2'},indent=2)+'\n')
        print('CREDIT_MATCHED_SUMMARY='+json.dumps(summary))


if __name__=='__main__':main()
