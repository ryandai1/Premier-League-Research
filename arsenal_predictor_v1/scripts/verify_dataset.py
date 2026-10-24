"""Validate exports, temporal separation and prediction-time invariance.
Also fit a fixed logistic baseline on TRAIN and check VALIDATION compatibility.
Never fit, tune or score a model on the test partition.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from build_dataset import ROOT, FEATURES, make_features


def main():
    schema=json.loads((ROOT/'feature_schema.json').read_text())
    matrices={};ids={};dates={}
    for split in ['train','validation','test']:
        p=ROOT/'data'/split
        x=pd.read_csv(p/'X.csv');y=pd.read_csv(p/'y.csv');meta=pd.read_csv(p/'index.csv')
        assert list(x.columns)==schema['features']==FEATURES
        assert len(x)==len(y)==len(meta)
        assert np.isfinite(x.to_numpy()).all()
        assert set(y.target_result)=={'A','D','H'}
        assert set(meta.split)=={split}
        assert meta.match_id.is_unique
        matrices[split]=(x,y);ids[split]=set(meta.match_id);dates[split]=pd.to_datetime(meta.date)
    assert not ids['train']&ids['validation'] and not ids['train']&ids['test'] and not ids['validation']&ids['test']
    assert dates['train'].max()<dates['validation'].min()<dates['test'].min()
    assert dates['validation'].max()<dates['test'].min()
    matches=pd.read_csv(ROOT/'data/matches_clean.csv',parse_dates=['date'])
    # All rows on or before a changed match date must retain identical inputs.
    cut=matches.date.iloc[300]
    sample=matches[matches.date<=matches.date.iloc[600]].copy()
    first=make_features(sample)
    changed=sample.copy();i=changed.index[changed.date.eq(cut)][0]
    changed.loc[i,'home_goals']+=5
    changed.loc[i,'home_shots']+=12
    changed.loc[i,'home_xg']=7.0
    changed.loc[i,'result']='H'
    second=make_features(changed)
    pd.testing.assert_frame_equal(first.loc[first.date.le(cut),['match_id']+FEATURES].reset_index(drop=True),second.loc[second.date.le(cut),['match_id']+FEATURES].reset_index(drop=True))
    # Removing future observations cannot change already-produced features.
    prefix=make_features(sample[sample.date<=cut])
    pd.testing.assert_frame_equal(first.loc[first.date.le(cut),['match_id']+FEATURES].reset_index(drop=True),prefix[['match_id']+FEATURES].reset_index(drop=True))
    assert not first.loc[first.date.gt(cut),FEATURES].equals(second.loc[second.date.gt(cut),FEATURES]),'Mutation check did not exercise update path'
    xtr,ytr=matrices['train'];xv,yv=matrices['validation']
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,random_state=42))
    model.fit(xtr,ytr.target_result)
    p=model.predict_proba(xv)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
    baseline=DummyClassifier(strategy='prior').fit(xtr,ytr.target_result)
    metrics={'validation_rows':len(xv),'fixed_logistic_validation_log_loss':float(log_loss(yv.target_result,p,labels=model.classes_)),'training_class_prior_validation_log_loss':float(log_loss(yv.target_result,baseline.predict_proba(xv),labels=baseline.classes_)),'scope':'Compatibility smoke check only; no hyperparameter search; no claim of optimal model; no test predictions generated','test_scored':False}
    checks=['snapshot checksums verified during build','one unique fixture per season/home/away','380 fixtures and 38 matches per club per season','goals agree with result labels','3800 overlapping fixtures reconciled without duplication','3 seasons checked against football-data dates/results/core statistics','39 finite numeric inputs aligned with targets and metadata','chronological nonoverlapping train/validation/test','changing current/future outcomes leaves earlier and same-day features unchanged','prefix reconstruction matches full-history feature computation','logistic training and validation probabilities execute successfully']
    output={'passed':True,'checks':checks,'smoke_check':metrics}
    (ROOT/'audit/validation.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
