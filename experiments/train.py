"""Prespecified, bounded comparison. Test results never select checkpoints."""
import copy, hashlib, json, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'arsenal_predictor_v1'
CLASSES=['A','D','H']
CONFIG={'seed':42,'max_epochs':100,'patience':20,'min_delta':0.0001,'lr':0.0001,'weight_decay':0.01,'batch_size':256,'hidden':[32,16],'dropout':0.1,'selection':'Arsenal validation log loss','threads':2}

def arsenal(frame):return (frame.home_team.eq('Arsenal')|frame.away_team.eq('Arsenal')).to_numpy()
def network(n):return nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,3))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load():
    schema=json.loads((DATA/'feature_schema.json').read_text()); splits={}
    for s in ['train','validation','test']:
        x=pd.read_csv(DATA/f'data/{s}/X.csv'); y=pd.read_csv(DATA/f'data/{s}/y.csv'); meta=pd.read_csv(DATA/f'data/{s}/index.csv')
        assert list(x)==schema['features'] and len(x)==len(y)==len(meta)
        splits[s]=(x.to_numpy(dtype=np.float32),y.target_result.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(),meta)
    return schema,splits

def fit(name,train,val,folder,features,initial=None):
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads']);torch.use_deterministic_algorithms(True)
    x,y=train; xv,yv=val
    model=network(x.shape[1]);scaler=StandardScaler()
    if initial:
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model_state'])
        scaler.mean_=np.array(saved['scaler_mean']);scaler.scale_=np.array(saved['scaler_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=len(features)
    else:scaler.fit(x)
    xt=torch.tensor(scaler.transform(x),dtype=torch.float32);yt=torch.tensor(y,dtype=torch.long)
    vt=torch.tensor(scaler.transform(xv),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    criterion=nn.CrossEntropyLoss();best=float('inf');wait=0;history=[];best_epoch=0
    # Fine-tuning may retain its initial checkpoint (epoch 0) if it never improves.
    if initial:
        model.eval()
        with torch.no_grad():best=float(criterion(model(vt),torch.tensor(yv,dtype=torch.long)))
    def save(epoch,loss):
        torch.save({'model_state':copy.deepcopy(model.state_dict()),'optimizer_state':opt.state_dict(),'epoch':epoch,'validation_log_loss':loss,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'features':features,'class_order':CLASSES,'config':CONFIG,'training_rows':len(x),'initial_checkpoint':str(initial) if initial else None},folder/f'{name}_best.pt')
    if initial:save(0,best)
    for epoch in range(1,CONFIG['max_epochs']+1):
        model.train();order=torch.randperm(len(xt));total=0.
        for start in range(0,len(xt),CONFIG['batch_size']):
            ids=order[start:start+CONFIG['batch_size']];opt.zero_grad();loss=criterion(model(xt[ids]),yt[ids]);loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():vl=float(criterion(model(vt),torch.tensor(yv,dtype=torch.long)))
        history.append({'epoch':epoch,'training_log_loss':total/len(xt),'validation_arsenal_log_loss':vl})
        # Save the true minimum; patience resets only on a material improvement.
        material=vl < best-CONFIG['min_delta']
        if vl<best:best=vl;best_epoch=epoch;save(epoch,vl)
        wait=0 if material else wait+1
        if wait>=CONFIG['patience']:break
    pd.DataFrame(history).to_csv(folder/f'{name}_history.csv',index=False)
    frozen=torch.load(folder/f'{name}_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(frozen['model_state']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    def predict(a):
        with torch.no_grad():return model(torch.tensor(scaler.transform(a),dtype=torch.float32)).softmax(1).numpy()
    assert best_epoch<=100 and len(history)<=100
    return predict,{'model':name,'best_epoch':best_epoch,'epochs_run':len(history),'validation_arsenal_log_loss':best,'training_rows':len(x),'checkpoint_sha256':digest(folder/f'{name}_best.pt')}

def score(y,p):
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    return {'log_loss':float(log_loss(y,p,labels=[0,1,2])),'brier_multiclass':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),'accuracy':float(accuracy_score(y,p.argmax(1))),'n':len(y)}

def run():
    schema,splits=load(); folder=ROOT/'results'/time.strftime('run_%Y%m%dT%H%M%S',time.gmtime());folder.mkdir(exist_ok=False)
    xt,yt,mt=splits['train']; xv,yv,mv=splits['validation']; a=arsenal(mt);av=arsenal(mv)
    inputs={f'{s}/{n}.csv':digest(DATA/f'data/{s}/{n}.csv') for s in splits for n in ['X','y','index']}
    (folder/'run_manifest.json').write_text(json.dumps({'config':CONFIG,'classes':CLASSES,'features':schema['features'],'input_sha256':inputs,'code_sha256':digest(__file__),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'single_seed':True},indent=2))
    predictors={};training=[]
    for name,mask in [('league_mlp',np.ones(len(xt),dtype=bool)),('arsenal_mlp',a)]:
        predictors[name],info=fit(name,(xt[mask],yt[mask]),(xv[av],yv[av]),folder,schema['features']);training.append(info)
    predictors['league_to_arsenal_mlp'],info=fit('league_to_arsenal_mlp',(xt[a],yt[a]),(xv[av],yv[av]),folder,schema['features'],folder/'league_mlp_best.pt');training.append(info)
    # Fixed credible baselines; no tuning on either validation or test.
    for name,mask in [('league_logistic',np.ones(len(xt),dtype=bool)),('arsenal_logistic',a)]:
        sc=StandardScaler().fit(xt[mask]);lr=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(sc.transform(xt[mask]),yt[mask])
        assert lr.classes_.tolist()==[0,1,2]
        predictors[name]=lambda x,sc=sc,lr=lr:lr.predict_proba(sc.transform(x))
        np.savez(folder/f'{name}.npz',mean=sc.mean_,scale=sc.scale_,coef=lr.coef_,intercept=lr.intercept_,classes=lr.classes_)
    prior=np.bincount(yt,minlength=3)/len(yt);predictors['league_prior']=lambda x:np.tile(prior,(len(x),1))
    np.save(folder/'league_prior.npy',prior)
    pd.DataFrame(training).to_csv(folder/'training_summary.csv',index=False)
    # Select the primary deployment candidate using Arsenal validation only.
    val_scores={name:score(yv[av],predict(xv[av]))['log_loss'] for name,predict in predictors.items()}
    selected=min(val_scores,key=val_scores.get)
    (folder/'selected_model.json').write_text(json.dumps({'selected_model':selected,'criterion':'Lowest Arsenal validation log loss; test unused','validation_scores':val_scores},indent=2))
    records=[];predictions={}
    for split in ['validation','test']:
        x,y,meta=splits[split]; am=arsenal(meta)
        for name,predict in predictors.items():
            p=predict(x);predictions[split,name]=p
            for population,mask in [('Arsenal',am),('Other EPL',~am),('All EPL',np.ones(len(y),dtype=bool))]:records.append({'model':name,'split':split,'population':population,**score(y[mask],p[mask])})
            table=meta.copy();table['target_class']=np.array(CLASSES)[y]
            for j,c in enumerate(CLASSES):table['p_'+c]=p[:,j]
            table.to_csv(folder/f'{name}_{split}_predictions.csv',index=False)
            ap=table.loc[am].copy();home=ap.home_team.eq('Arsenal');ap['p_win']=np.where(home,ap.p_H,ap.p_A);ap['p_draw']=ap.p_D;ap['p_loss']=np.where(home,ap.p_A,ap.p_H)
            ap.to_csv(folder/f'{name}_{split}_arsenal_probabilities.csv',index=False)
    metrics=pd.DataFrame(records);metrics.to_csv(folder/'metrics.csv',index=False)
    # Paired month-block bootstrap preserves within-month dependence; only 2 test seasons.
    x,y,meta=splits['test'];am=arsenal(meta);y=y[am];months=meta.loc[am,'date'].str[:7].to_numpy()
    lp=predictions['test','league_mlp'][am];ap=predictions['test','arsenal_mlp'][am]
    delta=-np.log(np.clip(lp[np.arange(len(y)),y],1e-12,1))+np.log(np.clip(ap[np.arange(len(y)),y],1e-12,1))
    rng=np.random.default_rng(42);unique=np.unique(months);boot=[]
    for _ in range(1000):
        ids=np.concatenate([np.flatnonzero(months==m) for m in rng.choice(unique,len(unique),replace=True)]);boot.append(float(delta[ids].mean()))
    uncertainty={'comparison':'league_mlp minus arsenal_mlp Arsenal test log loss; negative favors league','difference':float(delta.mean()),'month_block_bootstrap_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),'resamples':1000,'blocks':len(unique),'caveat':'Single training seed; short historical test; interval does not include training randomness.'}
    (folder/'comparison.json').write_text(json.dumps(uncertainty,indent=2));return folder

def predict_checkpoint(path,x):
    """Score prepared 39-column pre-match rows, in checkpoint feature order."""
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert list(x.columns)==ck['features'] and np.isfinite(x.to_numpy()).all()
    model=network(len(ck['features']));model.load_state_dict(ck['model_state']);model.eval()
    scaled=(x.to_numpy()-np.array(ck['scaler_mean']))/np.array(ck['scaler_scale'])
    with torch.no_grad():p=model(torch.tensor(scaled,dtype=torch.float32)).softmax(1).numpy()
    return pd.DataFrame(p,columns=ck['class_order'],index=x.index)
if __name__=='__main__':print(run())
