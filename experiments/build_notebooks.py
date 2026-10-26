"""Author the three reviewable notebooks; execution is a separate explicit command."""

from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def notebook(title):
    return [nb.v4.new_markdown_cell(title)]


def md(cells, text):
    cells.append(nb.v4.new_markdown_cell(text))


def code(cells, text):
    cells.append(nb.v4.new_code_cell(text))


def save(name, cells):
    n = nb.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    nb.validate(n)
    nb.write(n, ROOT / "notebooks" / name)


SETUP = """from pathlib import Path
import sys, json
PROJECT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'experiments/study.py').exists())
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from experiments import study, train
DATA=study.DATA; RUN=study.RUN
plt.rcParams.update({'font.size':11,'figure.figsize':(11,5),'axes.spines.top':False,'axes.spines.right':False})
BLUE='#236B8E'; GOLD='#B78126'; GREY='#737373'
def finish(name):
    plt.tight_layout()
    plt.savefig(RUN/'figures'/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.show()
"""


def build():
    c = notebook(
        """# 01 — Data sources, combinations, quality and exploratory visualization

Read this notebook **before modeling**. It creates a new, empty result directory; no prior model results are loaded. Then it inspects raw sources, reconstructs and audits the features, and profiles the frozen chronological splits.

**Reproducibility rerun after test inspection.** The historical test outcomes have already been seen. Descriptive inspection here does not make them an untouched holdout. Model settings and splits remain fixed; no hyperparameter choice is made from these plots.

The raw bundle remains frozen. Supplementary sources validate overlapping fixtures; they are **not concatenated as extra training samples**. Complete data tables behind the visualizations are exported alongside the figures."""
    )
    code(
        c,
        SETUP
        + """\nrecord=study.begin()
print('Fresh run:',record['run_id'],record['started_utc'])
print('Previous outputs loaded:',record['previous_outputs_loaded'])
sources=json.loads((DATA/'source_manifest.json').read_text())
raw=pd.read_csv(DATA/'raw/reza.csv',dtype={'Season':str})
print('Primary raw table:',raw.shape)
display(pd.DataFrame(sources)[['name','url','sha256']])""",
    )
    md(
        c,
        """## 1. What is combined—and what is not?
Reza is the primary 26-season match archive. Tara covers ten overlapping seasons and checks common fields. Football-data 2015/16, 2024/25 and 2025/26 snapshots provide three season spot-checks. They do not increase the fixture count. Overlap cannot establish fully independent upstream provenance: providers may share underlying sources.

Below, fixture identity is season + normalized home team + away team. The plotted diagonal is each source's row count; off-diagonal cells count shared fixtures. The source URLs and hashes above identify the exact snapshots; this execution makes no freshness claim.""",
    )
    code(
        c,
        """from experiments import audit
b=audit.b
canonical={'reza':b.canonical(raw,'reza'), 'tara':b.canonical(pd.read_csv(DATA/'raw/tara.csv'),'tara')}
for suffix in ['1516','2425','2526']:
    canonical['football_data_'+suffix]=b.canonical(pd.read_csv(DATA/f'raw/football_data_{suffix}.csv'),'fd',2000+int(suffix[:2]))
keys={name:set(map(tuple,frame[b.KEY].to_numpy())) for name,frame in canonical.items()}
primary=keys['reza']
source_summary=pd.DataFrame([{'source':name,'rows':len(frame),'columns_after_normalization':frame.shape[1], 'first_date':str(frame.date.min().date()),'last_date':str(frame.date.max().date()),'unique_fixtures':len(keys[name]),'overlap_with_primary':len(keys[name]&primary),'additional_fixtures':len(keys[name]-primary)} for name,frame in canonical.items()])
source_summary.to_csv(RUN/'data/source_summary.csv',index=False);display(source_summary)
overlap=pd.DataFrame({a:{b:len(keys[a]&keys[b]) for b in keys} for a in keys})
overlap.to_csv(RUN/'data/source_overlap.csv')
fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(overlap,cmap='Blues')
ax.set_xticks(range(len(keys)),list(keys),rotation=25,ha='right');ax.set_yticks(range(len(keys)),list(keys))
for i in range(len(keys)):
    for j in range(len(keys)):ax.text(j,i,str(overlap.iloc[i,j]),ha='center',va='center',color='white' if overlap.iloc[i,j]>5000 else 'black')
ax.set_title('Shared fixtures between frozen source snapshots');fig.colorbar(im,ax=ax,label='Fixtures')
finish('01_source_overlap')""",
    )
    md(
        c,
        """### Source disagreements and resolution
The one overlapping measurement disagreement is inspected before cleaning. No extra copy of the fixture is added. Reza's 15 home shots for Aston Villa–West Ham on 2015-12-26 is retained; the bundled football-data season corroborates it. The independent audit below checks that exact comparison.""",
    )
    code(
        c,
        """joined=canonical['reza'].merge(canonical['tara'],on=b.KEY,suffixes=('_reza','_tara'),validate='one_to_one')
conflicts=[]
for field in ['date','result',*b.MAP_T.values()]:
    for _,row in joined[joined[field+'_reza'].ne(joined[field+'_tara'])].iterrows():
        conflicts.append({**{key:row[key] for key in b.KEY},'field':field,'reza':row[field+'_reza'],'tara':row[field+'_tara']})
conflicts=pd.DataFrame(conflicts);conflicts.to_csv(RUN/'data/source_conflicts.csv',index=False);display(conflicts)
print('Primary fixtures retained:',len(primary),'Supplementary additions:',sum(len(v-primary) for k,v in keys.items() if k!='reza'))""",
    )
    md(
        c,
        """## 2. Raw coverage over time
Coverage is checked before trusting the source README. Every season should contain 380 matches and 20 clubs. Arsenal should contribute 38 appearances per full season. The raw archive includes measurements and end-of-season metadata; those columns are not automatically valid predictors.""",
    )
    code(
        c,
        """r=canonical['reza']
coverage=r.groupby('season_start').agg(matches=('result','size'),first_date=('date','min'),last_date=('date','max'))
coverage['Arsenal_matches']=r.assign(Arsenal=r.home_team.eq('Arsenal')|r.away_team.eq('Arsenal')).groupby('season_start').Arsenal.sum()
coverage.to_csv(RUN/'data/season_coverage.csv');display(coverage)
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
x=np.arange(len(coverage));labels=[f'{y}/{str(y+1)[-2:]}' for y in coverage.index]
axes[0].bar(x,coverage.matches,color=BLUE);axes[0].set(ylabel='EPL fixtures',title='Primary-source season coverage')
axes[1].bar(x,coverage.Arsenal_matches,color=GOLD);axes[1].set(ylabel='Arsenal fixtures',xlabel='Season')
axes[1].set_xticks(x,labels,rotation=60,ha='right');finish('02_season_coverage')""",
    )
    md(
        c,
        """## 3. Missingness and source-column inventory
Missing xG is not a zero measurement. xG availability changes by era and provider continuity is unverified, so xG is excluded from the fixed 39-feature model. Exported inventory lists every raw column, type, missing count and uniqueness. The chart highlights the largest missing fractions rather than hiding the rest.""",
    )
    code(
        c,
        """inventory=pd.DataFrame({'dtype':raw.dtypes.astype(str),'missing':raw.isna().sum(),'missing_percent':raw.isna().mean()*100,'unique_nonmissing':raw.nunique()})
inventory.to_csv(RUN/'data/raw_column_inventory.csv')
display(inventory.sort_values('missing_percent',ascending=False).head(20))
missing=inventory[inventory.missing>0].sort_values('missing_percent')
fig,ax=plt.subplots(figsize=(11,4));ax.barh(missing.index,missing.missing_percent,color=GOLD)
ax.set(xlim=(0,100),xlabel='Missing (%)',title=f'Largest raw missing fractions — {len(raw):,} matches, {raw.shape[1]} columns');finish('03_raw_missingness')
xg=r.groupby('season_start')[['home_xg','away_xg']].agg(lambda s:s.notna().mean()*100)
xg.to_csv(RUN/'data/xg_availability.csv')
xg.plot(marker='o',color=[BLUE,GOLD],figsize=(11,4));plt.ylim(0,105);plt.ylabel('Observed xG (%)');plt.xlabel('Season start year');plt.title('xG coverage by season — excluded from default model');finish('04_xg_coverage')""",
    )
    md(
        c,
        """## 4. Team coverage and raw outcome distributions
The training population is broader than Arsenal: promoted/relegated teams have unequal historical coverage. Source rows count matches; summing appearances counts each match twice. Home/away outcome categories are H/D/A; they become Arsenal win/draw/loss only after reorientation.""",
    )
    code(
        c,
        """appearances=pd.concat([r.home_team,r.away_team]).value_counts().sort_values()
appearances.rename('appearances').to_csv(RUN/'data/team_coverage.csv')
fig,ax=plt.subplots(figsize=(10,12));ax.barh(appearances.index,appearances,color=[GOLD if name=='Arsenal' else BLUE for name in appearances.index]);ax.set(xlabel='Match appearances',title='All clubs in the primary archive');finish('05_team_coverage')
fig,axes=plt.subplots(1,3,figsize=(13,4))
r.result.value_counts().reindex(['A','D','H']).plot.bar(ax=axes[0],color=BLUE,rot=0);axes[0].set(title='Home/away result classes',ylabel='Matches')
for ax,field in zip(axes[1:],['home_goals','away_goals']):
    ax.hist(r[field],bins=np.arange(r[field].max()+2)-.5,color=GOLD,edgecolor='white');ax.set(title=field.replace('_',' '),xlabel='Goals',ylabel='Matches')
finish('06_raw_outcomes')""",
    )
    md(
        c,
        """## 5. Reconstruct and audit before modeling
The audit checks raw hashes, source overlap, raw-to-clean equality, season counts, every reconstructed feature, all X/y/index cells, eligibility and chronological split membership. It also mutates later observations and verifies earlier features do not change.

Five inconsistent shot-stat team sides across three matches become missing; no result label is invented. End-of-season metadata repairs are logged separately and remain excluded from predictors. No source bytes are overwritten.""",
    )
    code(
        c,
        """verification=audit.run(RUN/'audit')
display(pd.Series(verification))
clean=pd.read_csv(DATA/'data/matches_clean.csv')
full=pd.read_csv(DATA/'data/prematch_all.csv')
shot_flags=clean[clean.home_shots_quality_flag|clean.away_shots_quality_flag]
display(shot_flags[['date','home_team','away_team','home_shots','home_sot','away_shots','away_sot']])
corrections=pd.read_csv(RUN/'audit/metadata_corrections.csv');display(corrections.groupby(['column','basis']).size().rename('cells').reset_index())""",
    )
    md(
        c,
        """## 8. Eligibility, fixed splits and class balance
Exclusion is an eligibility decision, not deletion from the archive. Missing lag histories can disproportionately exclude newly promoted teams and early-season fixtures. The chronological split is fixed before this rerun. Because historical tests were inspected earlier, these descriptive test distributions are disclosed as post-inspection EDA, not model-tuning evidence.""",
    )
    code(
        c,
        """split_order=['train','validation','test']
counts=pd.crosstab(full.split,full.core_ready).reindex(split_order).rename(columns={False:'Excluded',True:'Eligible'})
counts.to_csv(RUN/'data/eligibility.csv');display(counts)
counts.plot.bar(stacked=True,color=[GOLD,BLUE],rot=0);plt.ylabel('Matches');plt.title('Archived fixtures by modeling eligibility');finish('07_eligibility')
ready=full[full.core_ready];arsenal_mask=ready.home_team.eq('Arsenal')|ready.away_team.eq('Arsenal')
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,subset,title in [(axes[0],ready,'All eligible EPL'),(axes[1],ready[arsenal_mask],'Arsenal fixtures (home/away classes)')]:
    balance=pd.crosstab(subset.split,subset.target_result,normalize='index').reindex(split_order).reindex(columns=['A','D','H'])*100
    balance.to_csv(RUN/'data'/('class_balance_'+('Arsenal' if ax is axes[1] else 'EPL')+'.csv'))
    balance.plot.bar(ax=ax,color=[BLUE,GOLD,GREY],rot=0);ax.set(title=title,ylabel='Matches (%)',ylim=(0,100));ax.legend(title='Result')
finish('08_class_balance')
split_table=ready.groupby('split').agg(rows=('match_id','size'),first_date=('date','min'),last_date=('date','max')).reindex(split_order)
split_table['Arsenal_rows']=ready[arsenal_mask].groupby('split').size()
split_table.to_csv(RUN/'data/split_summary.csv');display(split_table)""",
    )
    md(
        c,
        """## 9. Feature construction and training-only distributions
The default inputs are 32 lagged performance means (home/away × 5/10-match windows × 8 statistics), four prior-appearance/rest variables, and three Elo variables. Histories use strictly earlier dates, with at least five valid observations; 10-match means may use fewer than ten when the minimum is met. Current-match goals, result, shots, and end-of-season standings are excluded as direct predictors.

Only **training rows** enter the following distribution/correlation plots. The full 39-column description and correlation matrix are exported. Small selected views keep the figures readable without dropping data from the exports.""",
    )
    code(
        c,
        """schema=json.loads((DATA/'feature_schema.json').read_text());features=schema['features']
xt=pd.read_csv(DATA/'data/train/X.csv');mt=pd.read_csv(DATA/'data/train/index.csv');am=mt.home_team.eq('Arsenal')|mt.away_team.eq('Arsenal')
feature_summary=xt.describe().T;feature_summary['missing']=xt.isna().sum();feature_summary.to_csv(RUN/'data/training_feature_summary.csv')
with pd.option_context('display.max_rows',None):display(feature_summary.round(3))
selected=['elo_home_minus_away','home_goals_for_mean5','away_goals_for_mean5','home_points_mean10','home_days_since_previous_epl','away_days_since_previous_epl']
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,field in zip(axes.flat,selected):
    bins=np.histogram_bin_edges(xt[field],bins=25)
    ax.hist(xt[field],bins=bins,density=True,histtype='step',linewidth=2,color=BLUE,label=f'EPL (n={len(xt)})')
    ax.hist(xt.loc[am,field],bins=bins,density=True,histtype='step',linestyle='--',linewidth=2,color=GOLD,label=f'Arsenal subset (n={am.sum()})')
    ax.set(title=field.replace('_',' '),ylabel='Density');ax.legend(fontsize=8)
finish('09_training_feature_distributions')
corr=xt.corr();corr.to_csv(RUN/'data/training_feature_correlations.csv')
chosen=selected+['home_shots_for_mean5','away_shots_for_mean5','home_sot_for_mean5','away_sot_for_mean5','home_elo','away_elo']
fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(corr.loc[chosen,chosen],cmap='RdBu_r',vmin=-1,vmax=1)
labels=[s.replace('_',' ') for s in chosen];ax.set_xticks(range(len(chosen)),labels,rotation=55,ha='right');ax.set_yticks(range(len(chosen)),labels);ax.set_title('Selected training-feature correlations — full matrix exported');fig.colorbar(im,ax=ax,label='Pearson correlation');finish('10_training_correlations')""",
    )
    md(
        c,
        """## Data review conclusion and handoff
The supplementary sources check the primary archive without expanding its training population. All modeling features and split files are independently reconstructed before fitting. Missing xG and end-of-season metadata are excluded from the default inputs; eligibility and team-coverage biases remain limitations.

Continue to notebook 02 only after the audit passes. The next notebook records initialization and every epoch. The original raw bundle's audit files are historical metadata; this notebook's `results/current/audit/` is the fresh verification.""",
    )
    code(
        c,
        """study.data_complete(verification)
print('Data stage complete:',study.manifest()['stage'])
print('Audit:',RUN/'audit/verification.json')
print('All data tables:',RUN/'data')""",
    )
    save("01_data_sources_and_audit.ipynb", c)

    c = notebook(
        """# 02 — Model development, controlled training and every epoch

Prerequisite: notebook 01 completed its data audit. **No old checkpoint, prediction or model result is available to this run.** All independent arms start from scratch; only optional fine-tuning loads the league checkpoint created here.

## Control, treatment and fixed settings
Control: Arsenal-only MLP on 815 training fixtures. Treatment: independent league-wide MLP on 7,977 training fixtures (including Arsenal). Optional extension: specialize the best league model on those same Arsenal training fixtures. Logistic models and a league class prior are reference controls.

39→32→16→3, ReLU, dropout 0.1, unweighted cross-entropy, AdamW learning rate 1e-4, weight decay 0.01, batch 256, seed 42, maximum 100 epochs, patience 20 and min_delta 1e-4. All checkpoints use the same 74 Arsenal validation fixtures. This compares practical pipelines under an epoch cap; the league gets more updates per epoch.

**Reproducibility rerun after test inspection.** No seed or setting is changed to manufacture different results. Seed resets make corresponding reruns deterministic; independent training arms need not have identical learned weights or predictions."""
    )
    code(
        c,
        SETUP
        + """\ndisplay(study.initialize_training())
display(pd.Series(train.CONFIG))
model=train.network(39)
print(model)
print('Trainable parameters:',sum(p.numel() for p in model.parameters()))
print('Objective: torch.nn.CrossEntropyLoss, raw logits, class order',train.CLASSES)""",
    )
    md(
        c,
        """## 1. Train the Arsenal-only control
The following output is generated live by the training loop: online training loss, end-of-epoch Arsenal validation loss/accuracy, selected checkpoint, patience and cumulative optimizer steps. Test scores are not computed per epoch.""",
    )
    code(c, """display(pd.Series(study.fit_stage('arsenal_mlp')))""")
    md(
        c,
        """## 2. Train the independent league challenger
This constructs a fresh model and a new scaler fitted only on league training rows. Resetting seed 42 gives the same initial tensors as the Arsenal model, but it does not copy its learned weights. Data, gradient updates and the resulting fitted model differ.""",
    )
    code(
        c,
        """display(pd.Series(study.fit_stage('league_mlp')))
display(pd.DataFrame(json.loads((study.MODELS/'frozen_validation_comparison.json').read_text())).T)""",
    )
    md(
        c,
        """## 3. Optional Arsenal specialization — enabled before evaluation
The independent comparison is frozen above. Fine-tuning starts from this run's best league checkpoint and retains its scaler, with a fresh AdamW optimizer. Its inherited validation loss is eligible as epoch zero. The training budget and stopping rules remain unchanged.""",
    )
    code(c, """display(pd.Series(study.fit_stage('league_to_arsenal_mlp')))""")
    md(
        c,
        """## 4. Fit fixed reference controls and freeze selection
Logistic regression uses C=1, max_iter=2000, training-only standardization and seed 42; the prior uses league training class frequencies. Selection includes all arms and uses Arsenal validation loss only.""",
    )
    code(
        c,
        """training,selection,checks=study.controls_and_freeze()
display(training.drop(columns='checkpoint_sha256'))
display(pd.Series(selection['validation_scores'],name='Arsenal validation log loss'))
print('Selected candidate:',selection['selected_model'])
display(pd.Series(checks))""",
    )
    md(
        c,
        """## 5. Learning curves and full epoch tables
Common loss scales make magnitudes comparable, but training populations differ and dropout is active only during training. The validation curves use identical fixtures. Best epochs are marked; reaching epoch 100 does not establish convergence.""",
    )
    code(
        c,
        """fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
all_history=[]
for ax,row in zip(axes,training.itertuples()):
    h=pd.read_csv(study.MODELS/f'{row.model}_history.csv');all_history.append(h.assign(model=row.model))
    ax.plot(h.epoch,h.training_log_loss,color=BLUE,label='Training')
    ax.plot(h.epoch,h.validation_arsenal_log_loss,color=GOLD,linestyle='--',label='Arsenal validation')
    ax.axvline(row.best_epoch,color='black',linestyle=':',label=f'Best {row.best_epoch}')
    ax.set(title=row.model.replace('_',' '),xlabel='Epoch');ax.legend(fontsize=8)
axes[0].set_ylabel('Cross-entropy loss');finish('11_learning_curves')
history=pd.concat(all_history,ignore_index=True);history.to_csv(study.MODELS/'all_epochs.csv',index=False)
lines=['# Every epoch from the fresh controlled execution','',f"Run ID: `{study.manifest()['run_id']}`. Test not scored per epoch. Training loss includes dropout; validation loss uses evaluation mode.",'','| Model | Epoch | Train loss | Arsenal validation loss | Validation accuracy | Best epoch | Patience | Updates |','|---|---:|---:|---:|---:|---:|---:|---:|']
for row in history.itertuples():
    lines.append(f'| {row.model} | {row.epoch} | {row.training_log_loss:.9f} | {row.validation_arsenal_log_loss:.9f} | {row.validation_arsenal_accuracy:.6f} | {row.best_epoch_so_far} | {row.patience_used} | {row.optimizer_steps} |')
(study.MODELS/'EVERY_EPOCH.md').write_text(chr(10).join(lines)+chr(10))
for name,group in history.groupby('model',sort=False):
    display(Markdown('### '+name))
    with pd.option_context('display.max_rows',None,'display.max_columns',None):display(group.drop(columns='model').round(9))""",
    )
    md(
        c,
        """## 6. Execution evidence
The run records timestamps, initial and selected-weight hashes, source/configuration hashes, transfer ancestry and optimizer updates. Equal initial hashes under seed 42 are expected. Different learned weights between the independent arms demonstrate that the two trained models have not been conflated. These checks verify implementation behavior; they do not establish statistical superiority.""",
    )
    code(
        c,
        """audits=pd.DataFrame({name:json.loads((study.MODELS/f'{name}_execution.json').read_text()) for name in study.NEURAL}).T
display(audits)
print('Total optimizer updates:',audits.optimizer_steps.sum())
print('No test metric file yet:',not (RUN/'evaluation/metrics.csv').exists())
assert not (RUN/'evaluation/metrics.csv').exists()
print('Stage:',study.manifest()['stage'])""",
    )
    save("02_models_and_training.ipynb", c)

    c = notebook(
        """# 03 — Frozen evaluation, results and interpretation

Prerequisite: notebook 02 completed all fits and froze validation-only selection. This notebook reloads the newly trained artifacts and computes test results once. **It does not compare repeated executions as separate treatments.**

Primary contrast: league MLP minus Arsenal-only MLP log loss on identical Arsenal test fixtures. Secondary metrics: multiclass Brier sum (0–2), accuracy and calibration. Test results have been inspected historically, so this remains a reproducibility rerun."""
    )
    code(
        c,
        SETUP
        + """\nmetrics,comparison,diagnostics=study.evaluate()
test=metrics.query("split=='test' and population=='Arsenal'")
display(test[['model','n','log_loss','brier_multiclass','accuracy']].round(6))
display(pd.Series(comparison))
display(pd.Series(diagnostics))""",
    )
    md(
        c,
        """## 1. One fresh result set: controls and treatments
The figure compares trained arms within this execution. It does not show three copies of deterministic rerun scores. Loss is primary; accuracy is secondary. Candidate selection remains based on validation.""",
    )
    code(
        c,
        """fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,metric,label in [(axes[0],'log_loss','Log loss (lower is better)'),(axes[1],'brier_multiclass','Brier sum (lower is better)')]:
    bars=ax.barh(test.model.str.replace('_',' '),test[metric],color=[GOLD if n=='arsenal_mlp' else BLUE for n in test.model]);ax.invert_yaxis()
    for bar,value in zip(bars,test[metric]):ax.text(value+.01,bar.get_y()+bar.get_height()/2,f'{value:.4f}',va='center',fontsize=9)
    ax.set(xlabel=label,xlim=(0,test[metric].max()*1.18))
fig.suptitle('Fresh evaluation — same 75 Arsenal test fixtures, 2024/25–2025/26');finish('12_test_metrics')""",
    )
    md(
        c,
        """## 2. Paired uncertainty
The month-block bootstrap preserves within-month grouping, uses 1,000 resamples and reports a 95% percentile interval. It measures test-sample variability, not variation across training seeds or new untouched periods. A negative difference favors the league MLP.""",
    )
    code(
        c,
        """lo,hi=comparison['interval_95_percentile'];delta=comparison['difference']
fig,ax=plt.subplots(figsize=(9,3));ax.errorbar(delta,0,xerr=[[delta-lo],[hi-delta]],fmt='o',color=BLUE,capsize=6);ax.axvline(0,color=GREY,linestyle='--')
ax.set(yticks=[0],yticklabels=['League − Arsenal MLP'],xlabel='Difference in Arsenal test log loss',title=f'Month-block 95% interval [{lo:.4f}, {hi:.4f}]');finish('13_paired_uncertainty')""",
    )
    md(
        c,
        """## 3. Calibration and outcome mistakes
Reliability bins are fixed at five intervals. Bin counts are shown in the table; 75 test fixtures make calibration estimates unstable. A confusion matrix uses the validation-selected candidate, with labels reoriented to Arsenal loss/draw/win.""",
    )
    code(
        c,
        """selection=json.loads((study.MODELS/'selected_model.json').read_text());selected=selection['selected_model']
fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].plot([0,1],[0,1],':',color=GREY,label='Perfect calibration')
calibration=[]
for name,color,marker in [('arsenal_mlp',GOLD,'s'),('league_mlp',BLUE,'o'),(selected,GREY,'^')]:
    p=pd.read_csv(RUN/'evaluation'/f'{name}_test_arsenal_probabilities.csv')
    p['win']=((p.home_team.eq('Arsenal')&p.target_class.eq('H'))|(p.away_team.eq('Arsenal')&p.target_class.eq('A'))).astype(int)
    p['bin']=pd.cut(p.p_win,np.linspace(0,1,6),include_lowest=True)
    grouped=p.groupby('bin',observed=True).agg(mean_prediction=('p_win','mean'),observed_rate=('win','mean'),count=('win','size'))
    axes[0].plot(grouped.mean_prediction,grouped.observed_rate,marker=marker,color=color,label=name.replace('_',' '));calibration.append(grouped.reset_index().assign(model=name))
axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean Arsenal win probability',ylabel='Observed win fraction',title='Arsenal win reliability');axes[0].legend(fontsize=8)
p=pd.read_csv(RUN/'evaluation'/f'{selected}_test_arsenal_probabilities.csv')
home=p.home_team.eq('Arsenal');actual=np.where(p.target_class.eq('D'),1,np.where((home&p.target_class.eq('H'))|(~home&p.target_class.eq('A')),2,0))
predicted=p[['p_loss','p_draw','p_win']].to_numpy().argmax(1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(actual,predicted,labels=[0,1,2]);axes[1].imshow(cm,cmap='Blues')
for i in range(3):
    for j in range(3):axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[1].set(xticks=range(3),yticks=range(3),xticklabels=['Loss','Draw','Win'],yticklabels=['Loss','Draw','Win'],xlabel='Predicted Arsenal outcome',ylabel='Actual Arsenal outcome',title='Validation-selected: '+selected.replace('_',' '));finish('14_calibration_confusion')
calibration=pd.concat(calibration,ignore_index=True);calibration.to_csv(RUN/'evaluation/calibration_bins.csv',index=False);display(calibration)""",
    )
    md(
        c,
        """## 4. Temporal/population transfer and per-fixture probabilities
Non-Arsenal test fixtures are later matches from the same competition. This is not external-league validation. Full A/D/H and Arsenal-oriented prediction files are exported for every model. The following table shows the last ten historical Arsenal fixtures for the validation-selected candidate.""",
    )
    code(
        c,
        """display(metrics.query("split=='test' and population=='Other EPL'")[['model','n','log_loss','brier_multiclass','accuracy']].round(6))
display(p[['date','home_team','away_team','target_class','p_win','p_draw','p_loss']].tail(10).round(4))""",
    )
    md(
        c,
        """## 5. What the fresh execution establishes
Data provenance and features were re-audited; models trained from scratch with recorded epochs; only fine-tuning loaded a same-run checkpoint. The independent arms have distinct fitted weights and predictions, and their outputs respond to input-row changes. No cached result was used to produce metrics.

Repeating this deterministic pipeline with unchanged seed/data/settings can still reproduce earlier values. Equality across corresponding reruns is different from equality between the Arsenal and league arms. This study does not change the fixed protocol to create numerical variation.

Limitations: one seed, inspected test outcomes, unequal update counts per epoch, eligibility selection, short test span, uncertain original-observer lineage and no external competition. A new untouched period is required for a new confirmatory claim.""",
    )
    code(
        c,
        """summary={'run_id':study.manifest()['run_id'],'label':study.manifest()['label'],'selected_model':selected,'arsenal_test_results':test.to_dict('records'),'primary_comparison':comparison,'implementation_checks':diagnostics}
study.write(RUN/'evaluation/summary.json',summary)
print('Completed fresh run:',summary['run_id'])
print('Primary contrast:',comparison['difference'])
print('Validation-selected candidate:',selected)
print('Current results:',RUN)""",
    )
    save("03_evaluation_and_results.ipynb", c)


if __name__ == "__main__":
    build()
