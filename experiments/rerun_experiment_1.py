"""Execute unchanged experiment 1, observing epochs without changing its training code."""
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from experiments import train


def run():
    reference = train.ROOT / 'results/run_20261024T140407'
    experiment2 = train.ROOT / 'results/experiment_2/run_20261026T002413'
    manifest = json.loads((reference / 'run_manifest.json').read_text())
    assert train.digest(train.__file__) == manifest['code_sha256']
    assert train.CONFIG == manifest['config']
    for name, expected in manifest['input_sha256'].items():
        assert train.digest(train.DATA / 'data' / name) == expected
    started = datetime.now(timezone.utc).isoformat()
    events = []
    # Observe the original loop after its checkpoint/patience update.
    # Reading local variables consumes no RNG and changes no tensor or optimizer.
    source, first_line = inspect.getsourcelines(train.fit)
    epoch_end_line = first_line + next(i for i, s in enumerate(source) if "if wait>=CONFIG['patience']:break" in s)
    def observe(frame, event, arg):
        if frame.f_code is not train.fit.__code__:
            return None
        if event == 'line' and frame.f_lineno == epoch_end_line:
            v = frame.f_locals
            row = {'model': v['name'], **v['history'][-1], 'best_epoch_so_far': v['best_epoch'],
                   'patience_used': v['wait'], 'observed_utc': datetime.now(timezone.utc).isoformat(),
                   'optimizer_steps': max(int(s['step']) for s in v['opt'].state.values())}
            events.append(row)
            print(f"{row['model']} epoch={row['epoch']:03d} train={row['training_log_loss']:.9f} validation={row['validation_arsenal_log_loss']:.9f} best_epoch={row['best_epoch_so_far']} patience={row['patience_used']}/20 steps={row['optimizer_steps']}", flush=True)
        return observe
    prior_trace = sys.gettrace()
    try:
        sys.settrace(observe)
        folder = train.run()  # Original league → Arsenal → fine-tuning runner, unchanged.
    finally:
        sys.settrace(prior_trace)
    finished = datetime.now(timezone.utc).isoformat()
    epochs = pd.DataFrame(events)
    epochs.to_csv(folder/'observed_epochs.csv', index=False)
    assert list(epochs.model.unique()) == ['league_mlp','arsenal_mlp','league_to_arsenal_mlp']
    rows = []
    for label, old in [('experiment_1_original', reference), ('experiment_2_latest', experiment2)]:
        for model in pd.read_csv(folder/'metrics.csv').model.unique():
            for split in ['validation','test']:
                fresh = pd.read_csv(folder/f'{model}_{split}_predictions.csv')
                prior = pd.read_csv(old/f'{model}_{split}_predictions.csv')
                pd.testing.assert_frame_equal(fresh.drop(columns=['p_A','p_D','p_H']),prior.drop(columns=['p_A','p_D','p_H']))
                difference = float(np.abs(fresh[['p_A','p_D','p_H']].to_numpy()-prior[['p_A','p_D','p_H']].to_numpy()).max())
                rows.append({'reference':label,'model':model,'split':split,'max_probability_difference':difference})
    pd.DataFrame(rows).to_csv(folder/'prediction_comparison.csv', index=False)
    weight_checks = []
    for model in epochs.model.unique():
        new_ck = torch.load(folder/f'{model}_best.pt', weights_only=False)
        for label, old in [('experiment_1_original',reference),('experiment_2_latest',experiment2)]:
            old_ck = torch.load(old/f'{model}_best.pt', weights_only=False)
            same = all(torch.equal(value,old_ck['model_state'][name]) for name,value in new_ck['model_state'].items())
            old_h = pd.read_csv(old/f'{model}_history.csv')
            new_h = pd.read_csv(folder/f'{model}_history.csv')
            cols = ['epoch','training_log_loss','validation_arsenal_log_loss']
            history_same = new_h[cols].equals(old_h[cols])
            weight_checks.append({'reference':label,'model':model,'weights_exactly_equal':same,'epoch_losses_exactly_equal':history_same})
        observed = epochs[epochs.model.eq(model)]
        assert observed.epoch.tolist() == new_h.epoch.tolist()
        assert observed.training_log_loss.tolist() == new_h.training_log_loss.tolist() or np.allclose(observed.training_log_loss,new_h.training_log_loss,rtol=0,atol=1e-15)
    pd.DataFrame(weight_checks).to_csv(folder/'weights_and_epochs_comparison.csv',index=False)
    summary = {'label':'Experiment 1 reproducibility rerun after test inspection',
               'started_utc':started,'finished_utc':finished,
               'original_training_source_sha256':train.digest(train.__file__),
               'rerun_observer_source_sha256':train.digest(__file__),
               'original_training_source_unchanged':True,
               'execution_order':epochs.model.unique().tolist(),
               'observed_epochs':len(events),
               'optimizer_updates':int(epochs.groupby('model').optimizer_steps.max().sum()),
               'max_probability_difference':max(r['max_probability_difference'] for r in rows),
               'all_weights_exactly_equal':all(r['weights_exactly_equal'] for r in weight_checks),
               'all_epoch_losses_exactly_equal':all(r['epoch_losses_exactly_equal'] for r in weight_checks)}
    (folder/'rerun_verification.json').write_text(json.dumps(summary,indent=2))
    report = ['# Experiment 1 fresh rerun — every observed epoch\n',
              'Reproducibility rerun after test inspection. Original training source is unchanged; a Python trace observer records epoch results without changing RNG, gradients, or optimizer state.\n',
              'Order: league from scratch → Arsenal from scratch → league checkpoint fine-tuned on Arsenal.\n',
              '## Verification\n', '```json',json.dumps(summary,indent=2),'```',
              '## Epoch results\n','| Model | Epoch | Train loss | Arsenal validation loss | Best epoch | Patience | Updates |','|---|---:|---:|---:|---:|---:|---:|']
    for r in epochs.itertuples():
        report.append(f'| {r.model} | {r.epoch} | {r.training_log_loss:.9f} | {r.validation_arsenal_log_loss:.9f} | {r.best_epoch_so_far} | {r.patience_used} | {r.optimizer_steps} |')
    (folder/'RERUN_REPORT.md').write_text('\n'.join(report))
    print(json.dumps(summary,indent=2))
    return folder

if __name__ == '__main__':
    print(run())
