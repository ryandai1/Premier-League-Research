"""Reload a frozen run's selected artifact; feature construction stays separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.train import predict_checkpoint

def predict(run, features, model=None):
    run=Path(run);manifest=json.loads((run/'run_manifest.json').read_text())
    if list(features.columns)!=manifest['features']:raise ValueError('Feature order differs from the frozen run.')
    if not np.isfinite(features.to_numpy()).all():raise ValueError('Inputs must be finite, eligible pre-match features.')
    model=model or json.loads((run/'selected_model.json').read_text())['selected_model']
    if model.endswith('mlp'):return predict_checkpoint(run/f'{model}_best.pt',features)
    if model=='league_prior':p=np.tile(np.load(run/'league_prior.npy'),(len(features),1))
    else:
        with np.load(run/f'{model}.npz') as c:
            assert c['classes'].tolist()==[0,1,2]
            logits=((features.to_numpy()-c['mean'])/c['scale'])@c['coef'].T+c['intercept']
            logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
    return pd.DataFrame(p,columns=manifest['classes'],index=features.index)

def arsenal_probabilities(probabilities,metadata):
    if not probabilities.index.equals(metadata.index):raise ValueError('Probability and fixture indexes must align.')
    mask=metadata.home_team.eq('Arsenal')|metadata.away_team.eq('Arsenal')
    out=metadata.loc[mask].copy();p=probabilities.loc[mask];home=out.home_team.eq('Arsenal')
    out['p_win']=np.where(home,p.H,p.A);out['p_draw']=p.D;out['p_loss']=np.where(home,p.A,p.H)
    return out
