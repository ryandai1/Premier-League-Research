"""Build the active, isolated-data notebook sequence; no training on import."""
from pathlib import Path
import nbformat as nb
ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys, json, os
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'experiments/isolated_arsenal.py').exists())
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, joblib
from IPython.display import display, Markdown, Image
from experiments import train
from experiments.isolated_arsenal import OUTPUT
RUN=OUTPUT
REVIEW=ROOT/'results/arsenal_isolated_review'
(REVIEW/'figures').mkdir(parents=True,exist_ok=True)
record=json.loads((RUN/'manifest.json').read_text())
for rel,digest in record['artifact_sha256'].items():
    assert train.digest(RUN/rel)==digest,rel
features=record['features']
plt.rcParams.update({'figure.figsize':(11,5),'axes.spines.top':False,'axes.spines.right':False})
def finish(name):
    plt.tight_layout(); plt.savefig(REVIEW/'figures'/f'{name}.png',dpi=110,bbox_inches='tight'); plt.show(); plt.close()
def load(population,split):
    folder=RUN/population/split
    return (pd.read_csv(folder/'X.csv').astype(np.float32),pd.read_csv(folder/'y.csv').target_result.map({'A':0,'D':1,'H':2}).to_numpy(),pd.read_csv(folder/'index.csv'))
def predict(name,x):
    if name.endswith('mlp'):
        return train.predict_checkpoint(RUN/'models'/f'{name}_best.pt',x).to_numpy()
    return joblib.load(RUN/'models'/f'{name}.joblib').predict_proba(x)
metrics=pd.read_csv(RUN/'evaluation/metrics.csv')
names=metrics.model.drop_duplicates().tolist()
# Source duplicate is a deterministic replication, excluded from model selection.
candidates=[n for n in names if n!='league_frozen_source_mlp']
validation=metrics.query("split=='validation'").set_index('model').loc[candidates]
selected=validation.log_loss.idxmin()
'''
def save(name, title, sections):
    cells=[nb.v4.new_markdown_cell(title),nb.v4.new_code_cell(SETUP)]
    for kind,text in sections: cells.append(nb.v4.new_code_cell(text) if kind=='code' else nb.v4.new_markdown_cell(text))
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nb.write(n,ROOT/'notebooks'/name)

def build():
    save('01_data_sources_and_audit.ipynb','# 01 — Corrected data sources and isolation\n\nActive model inputs are `results/arsenal_isolated/{league,arsenal}/`. The frozen original bundle is source evidence, not the league model input. These cells audit existing data without training. All dates retain the original chronological partitions.',[
    ('markdown','## Raw sources and overlap\nSupplementary sources cross-check shared fixtures; they add no training rows. Raw archive plots below describe input evidence before Arsenal exclusion.'),
    ('code', '''sources=pd.read_csv(ROOT/'results/current/data/source_summary.csv')
display(sources)
overlap=pd.read_csv(ROOT/'results/current/data/source_overlap.csv',index_col=0)
fig,ax=plt.subplots(); im=ax.imshow(overlap.to_numpy(),cmap='Blues');ax.set_xticks(range(len(overlap)),overlap.columns,rotation=30);ax.set_yticks(range(len(overlap)),overlap.index)
for i in range(len(overlap)):
    for j in range(len(overlap)): ax.text(j,i,str(overlap.iloc[i,j]),ha='center',va='center')
ax.set_title('Frozen-source overlap — not additional training samples');finish('01_source_overlap')
raw=pd.read_csv(train.DATA/'data/matches_clean.csv')
raw.groupby('season_start').size().plot.bar();plt.ylabel('Fixtures');plt.title('Raw archive season coverage');finish('02_season_coverage')
raw.isna().mean().sort_values().tail(12).mul(100).plot.barh();plt.xlabel('Missing (%)');finish('03_missingness')
raw.groupby('season_start')[['home_xg','away_xg']].agg(lambda x:x.notna().mean()*100).plot();plt.ylabel('Observed xG (%)');plt.title('xG excluded from the 39 model features');finish('04_xg_coverage')
pd.concat([raw.home_team,raw.away_team]).value_counts().sort_values().plot.barh(figsize=(9,11));plt.xlabel('Raw fixture appearances');finish('05_team_coverage')
fig,axs=plt.subplots(1,2);raw.result.value_counts().reindex(['A','D','H']).plot.bar(ax=axs[0]);raw[['home_goals','away_goals']].plot.hist(bins=10,alpha=.5,ax=axs[1]);finish('06_raw_outcomes')'''),
    ('markdown','## Corrected splits and feature reconstruction\nArsenal fixtures are removed before source histories and Elo are computed. League: 7,142/654/656; Arsenal: 815/74/75. Source checkpoints use only non-Arsenal validation. Target features retain actual strictly earlier histories, creating a source/target distribution difference. Earlier observed evaluation-period matches may update pre-match histories, never model weights.'),
    ('code','''from experiments.isolated_arsenal import league_features
clean=pd.read_csv(train.DATA/'data/matches_clean.csv',parse_dates=['date'])
rebuilt=league_features(clean)
ready=rebuilt[features].notna().all(axis=1)
identities=[]; counts=[]; balances=[]
for pop in ['league','arsenal']:
    for split in ['train','validation','test']:
        x,y,meta=load(pop,split)
        assert train.arsenal(meta).all() if pop=='arsenal' else not train.arsenal(meta).any()
        assert meta.match_id.is_unique and np.isfinite(x).all().all()
        identities.append(set(meta.match_id));counts.append({'population':pop,'split':split,'n':len(y),'first':meta.date.min(),'last':meta.date.max()})
        balances.append(pd.Series(np.bincount(y,minlength=3)/len(y),index=['A','D','H'],name=f'{pop}/{split}'))
        if pop=='league':
            period={'train':rebuilt.season_start<=2021,'validation':rebuilt.season_start.between(2022,2023),'test':rebuilt.season_start>=2024}[split]
            expected=rebuilt.loc[ready & period]
            assert expected.match_id.tolist()==meta.match_id.tolist()
            np.testing.assert_array_equal(expected[features].to_numpy(dtype=np.float32),x.to_numpy())
for i,ids in enumerate(identities): assert all(not ids & previous for previous in identities[:i])
counts=pd.DataFrame(counts);display(counts)
counts.pivot(index='split',columns='population',values='n').reindex(['train','validation','test']).plot.bar();plt.ylabel('Eligible fixtures');finish('07_corrected_splits')
pd.DataFrame(balances).plot.bar(stacked=True);plt.ylabel('Class proportion');finish('08_class_balance')
display(pd.read_csv(RUN/'excluded_league_fixtures.csv').head(10))'''),
    ('code','''xl,_,_=load('league','train'); xa,_,_=load('arsenal','train')
chosen=['home_goals_for_mean5','away_goals_for_mean5','home_elo','away_elo','elo_home_minus_away','home_days_since_previous_epl']
fig,axs=plt.subplots(2,3,figsize=(13,7))
for ax,f in zip(axs.flat,chosen):
    ax.hist(xl[f],bins=25,density=True,alpha=.5,label='League excluding Arsenal');ax.hist(xa[f],bins=25,density=True,alpha=.5,label='Arsenal');ax.set_title(f)
axs.flat[0].legend(fontsize=8);finish('09_training_distributions')
fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(xl[chosen].corr(),vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(6),chosen,rotation=75);ax.set_yticks(range(6),chosen);fig.colorbar(im);finish('10_source_training_correlations')
print('Reconstructed source inputs match every saved eligible feature row.')''')])
    save('02_models_and_training.ipynb','# 02 — Corrected models and training\n\nThe saved corrected experiment is complete. Default execution reviews its recorded training; the explicit opt-in cell below runs a fresh experiment from this notebook. It never silently reuses old checkpoints for fresh pretraining.',[
    ('markdown','## Optional fresh training\nSet `RUN_NEW_EXPERIMENT=True` and choose a new directory to train all seven arms. Both source fits start from scratch. Existing saved experiments are never overwritten. To review that rerun in notebooks 01–04, change their `RUN` assignment to that new directory.'),
    ('code', '''RUN_NEW_EXPERIMENT=False
NEW_OUTPUT=ROOT/'work/arsenal_isolated_notebook_rerun'
if RUN_NEW_EXPERIMENT:
    from experiments.isolated_arsenal import run
    run(NEW_OUTPUT)
else:
    print('Reviewing completed training; no fresh fit requested.')
summary=pd.read_csv(RUN/'models/training_summary.csv');display(summary)
display(pd.Series(record['config']))
audits=pd.DataFrame([json.loads((RUN/'models'/f'{n}_execution.json').read_text()) for n in summary.model]);display(audits)'''),
    ('markdown','## Learning curves\nSource loss curves use non-Arsenal validation; control/adaptation curves use Arsenal validation. Their validation values describe different populations and should not be ranked against each other. Online training loss includes dropout.'),
    ('code','''fig,axs=plt.subplots(2,3,figsize=(16,9)); histories=[]
for ax,row in zip(axs.flat,summary.itertuples()):
    h=pd.read_csv(RUN/'models'/f'{row.model}_history.csv');histories.append(h.assign(model=row.model))
    ax.plot(h.epoch,h.training_log_loss,label='Training');ax.plot(h.epoch,h.validation_log_loss,label=f'{row.validation_population} validation');ax.axvline(row.best_epoch,ls=':',color='grey');ax.set_title(row.model);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
axs.flat[-1].axis('off');finish('11_learning_curves')
with pd.option_context('display.max_rows',None): display(pd.concat(histories,ignore_index=True))'''),
    ('code','''def ck(name): return torch.load(RUN/'models'/f'{name}_best.pt',weights_only=False,map_location='cpu')
a=ck('league_frozen_source_mlp'); b=ck('head_finetune_mlp')
checks={key:torch.equal(a['model_state'][key],b['model_state'][key]) for key in ['0.weight','0.bias','3.weight','3.bias']}
assert all(checks.values())
assert audits.set_index('model').loc['head_finetune_mlp','trainable_parameters']==['5.weight','5.bias']
assert any(not torch.equal(a['model_state'][key],b['model_state'][key]) for key in ['5.weight','5.bias'])
display(pd.Series(checks,name='frozen_hidden_parameter_unchanged'))
print('Head-only adaptation changed output parameters. Both independent source fits completed with identical deterministic settings.')''')])
    evaluation=[('code','''for split in ['validation','test']:
    x,y,meta=load('arsenal',split)
    for name in names:
        p=predict(name,x); saved=pd.read_csv(RUN/'evaluation'/f'{name}_{split}_predictions.csv')
        assert saved.match_id.tolist()==meta.match_id.tolist()
        np.testing.assert_allclose(saved[['p_A','p_D','p_H']].to_numpy().astype(p.dtype),p,rtol=0,atol=1e-15)
        row=metrics.query('model==@name and split==@split').iloc[0]
        for metric,value in train.score(y,p).items(): np.testing.assert_allclose(value,row[metric],rtol=0,atol=1e-12)
display(metrics)
print('Validation-selected candidate for diagnostics:',selected)
print('Selection is retrospective within this fixed candidate set; test does not choose it.')
fig,axs=plt.subplots(1,3,figsize=(16,6));test=metrics.query("split=='test'").set_index('model').loc[candidates]
for ax,m in zip(axs,['log_loss','brier_multiclass','accuracy']): test[m].plot.barh(ax=ax);ax.set_title(m+(' (higher better)' if m=='accuracy' else ' (lower better)'))
finish('12_test_metrics')'''),('markdown','## Paired uncertainty\nLeague-minus-Arsenal-only MLP test log loss, paired by fixture and resampled by month (1,000 bootstrap replicates, seed 42). This measures test-sample variability, not training-seed variability. Negative favors league pretraining. The test period was previously inspected.'),('code','''x,y,meta=load('arsenal','test')
a=predict('arsenal_mlp',x);b=predict('league_mlp',x)
delta=-np.log(np.clip(b[np.arange(len(y)),y],1e-15,1))+np.log(np.clip(a[np.arange(len(y)),y],1e-15,1))
months=meta.date.str[:7].to_numpy();groups=np.unique(months);rng=np.random.default_rng(42)
boot=[np.concatenate([delta[months==m] for m in rng.choice(groups,size=len(groups),replace=True)]).mean() for _ in range(1000)]
lo,hi=np.quantile(boot,[.025,.975]);point=delta.mean()
comparison={'contrast':'league minus Arsenal-only test log loss','difference':float(point),'interval_95':[float(lo),float(hi)],'replicates':1000}
(REVIEW/'comparison.json').write_text(json.dumps(comparison,indent=2))
display(pd.Series(comparison))
fig,ax=plt.subplots(figsize=(9,3));ax.hlines(0,lo,hi);ax.plot(point,0,'o');ax.axvline(0,color='grey',ls='--');ax.set_xlabel('Paired log-loss difference (league minus Arsenal-only)');ax.set_yticks([]);finish('13_paired_uncertainty')'''),('markdown','## Calibration and confusion\nFive fixed bins show Arsenal win probability versus observed wins for the validation-selected candidate and dummy. Empty bins are omitted; bin counts are exported. Confusion uses A/D/H match outcomes.'),('code','''from sklearn.metrics import confusion_matrix
fig,axs=plt.subplots(1,2,figsize=(13,5));axs[0].plot([0,1],[0,1],':',color='grey');cal=[]
home=meta.home_team.eq('Arsenal').to_numpy();won=np.where(home,y==2,y==0)
for name in [selected,'league_dummy']:
    p=predict(name,x);win=np.where(home,p[:,2],p[:,0]);bin_id=np.minimum((win*5).astype(int),4)
    table=pd.DataFrame({'bin':bin_id,'probability':win,'won':won}).groupby('bin').agg(mean_prediction=('probability','mean'),observed_rate=('won','mean'),n=('won','size')).reset_index();table['model']=name;cal.append(table)
    axs[0].plot(table.mean_prediction,table.observed_rate,'o-',label=name)
axs[0].set(xlabel='Predicted Arsenal win probability',ylabel='Observed win fraction',xlim=(0,1),ylim=(0,1));axs[0].legend()
p=predict(selected,x);cm=confusion_matrix(y,p.argmax(1),labels=[0,1,2]);axs[1].imshow(cm,cmap='Blues')
for i in range(3):
    for j in range(3): axs[1].text(j,i,str(cm[i,j]),ha='center',va='center')
axs[1].set(xticks=range(3),yticks=range(3),xticklabels=['A','D','H'],yticklabels=['A','D','H'],xlabel='Predicted',ylabel='Actual',title=selected)
finish('14_calibration_confusion');cal=pd.concat(cal);cal.to_csv(REVIEW/'calibration_bins.csv',index=False);display(cal)'''),('markdown','## Interpretation\nThese corrected arms use isolated source data. Full and head-only fine-tuning make tiny changes in this run. Accuracy and probability scores can rank models differently. No other target teams were trained. Non-Arsenal target-transfer results from the old study are not reused: that is a different evaluation population and not needed for the requested Arsenal experiments.')]
    save('03_evaluation_and_results.ipynb','# 03 — Corrected frozen evaluation and graphics\n\nEvery model is reloaded from `results/arsenal_isolated/models/`; all scores are recomputed on the corrected Arsenal validation/test fixtures. No model fitting occurs here.',evaluation)
    save('04_arsenal_isolated_experiments.ipynb','# 04 — Corrected experiment overview\n\nNotebooks 01–03 now use this same experiment. This overview includes their graphics and a compact score table. Run 01–03 first to regenerate graphics. Training remains in the explicit cell of notebook 02.',[
    ('code',"display(pd.DataFrame(record['rows']));display(metrics);display(pd.read_csv(RUN/'models/training_summary.csv'))"),
    ('code',"for name in ['07_corrected_splits','11_learning_curves','12_test_metrics','13_paired_uncertainty','14_calibration_confusion']:\n    path=REVIEW/'figures'/f'{name}.png'\n    assert path.exists(), 'Run notebooks 01–03 first'\n    display(Image(filename=str(path)))"),
    ('markdown','Source training excludes Arsenal before feature generation and source checkpoint selection excludes Arsenal. Source and target historical feature distributions differ by design. Every requested model arm has completed; head-only hidden weights are checked in 02. These are exploratory, single-seed results on an already inspected test period.')])
    save('Frozen_Model_Test_Review.ipynb','# Frozen model test review — corrected Arsenal-excluded experiment\n\nInference only, using the new checkpoints. Every model receives the same 75 Arsenal test fixtures. The selected model is chosen by Arsenal validation log loss. This replaces the old review of the inclusive models.',[
    ('code','''x,y,meta=load('arsenal','test')
actual=pd.read_csv(train.DATA/'data/matches_clean.csv')[['match_id','home_goals','away_goals']]
base=meta.merge(actual,on='match_id',how='left',validate='one_to_one');assert base.home_goals.notna().all()
folder=REVIEW/'test_review';folder.mkdir(exist_ok=True)
base.to_csv(folder/'arsenal_test_fixtures.csv',index=False)
print('Validation-selected model:',selected);display(base)
tables={};scores=[]
for name in names:
    p=predict(name,x);pred=p.argmax(1);home=base.home_team.eq('Arsenal').to_numpy()
    t=base.copy();t['actual_result']=np.array(['A','D','H'])[y];t['predicted_result']=np.array(['A','D','H'])[pred];t['correct']=pred==y
    t['actual_arsenal_result']=np.where(y==1,'Draw',np.where(np.where(home,y==2,y==0),'Win','Loss'))
    t['predicted_arsenal_result']=np.where(pred==1,'Draw',np.where(np.where(home,pred==2,pred==0),'Win','Loss'))
    t['p_win']=np.where(home,p[:,2],p[:,0]);t['p_draw']=p[:,1];t['p_loss']=np.where(home,p[:,0],p[:,2]);t['confidence']=p.max(1)
    t.to_csv(folder/f'{name}_arsenal_test_review.csv',index=False);tables[name]=t
    scores.append({'model':name,**train.score(y,p)})
    expected=metrics.query("model==@name and split=='test'").iloc[0]
    for metric,value in train.score(y,p).items(): np.testing.assert_allclose(value,expected[metric],atol=1e-12,rtol=0)
scores=pd.DataFrame(scores);scores.to_csv(folder/'metrics.csv',index=False);display(scores)'''),
    ('code',"for name,table in tables.items():\n    display(Markdown('### '+name))\n    with pd.option_context('display.max_rows',None): display(table)"),
    ('code',"for rel,digest in record['artifact_sha256'].items(): assert train.digest(RUN/rel)==digest\nprint('All source experiment artifacts remain unchanged; all seven model verdict tables exported.')")])
if __name__=='__main__': build()
