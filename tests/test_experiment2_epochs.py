"""Validate the executed epoch trace, optimizer state and checkpoint ancestry."""
import json
import math
import unittest
from pathlib import Path
import pandas as pd
import torch
from experiments.epoch_training import state_digest
from experiments.train import network

ROOT = Path(__file__).resolve().parents[1]

class EpochTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = sorted(p.parent for p in (ROOT/'results/experiment_2').glob('run_*/all_epochs.csv'))[-1]

    def test_patience_and_checkpoint_selection(self):
        for name in ['arsenal_mlp', 'league_mlp', 'league_to_arsenal_mlp']:
            with self.subTest(model=name):
                audit = json.loads((self.run_dir/f'{name}_execution.json').read_text())
                history = pd.read_csv(self.run_dir/f'{name}_history.csv')
                ck = torch.load(self.run_dir/f'{name}_best.pt', weights_only=False)
                best = float('inf')
                if audit['initial_checkpoint']:
                    best = torch.load(self.run_dir/'league_mlp_best.pt', weights_only=False)['validation_log_loss']
                wait = 0
                for row in history.itertuples():
                    material = row.validation_arsenal_log_loss < best - 1e-4
                    self.assertEqual(row.checkpoint_saved, row.validation_arsenal_log_loss < best)
                    best = min(best, row.validation_arsenal_log_loss)
                    wait = 0 if material else wait+1
                    self.assertEqual(row.patience_used, wait)
                self.assertLessEqual(len(history),100)
                self.assertTrue(len(history)==100 or wait==20)
                self.assertEqual(ck['epoch'], audit['best_epoch'])
                self.assertAlmostEqual(ck['validation_log_loss'], best)
                steps_per_epoch = math.ceil(audit['training_rows']/256)
                self.assertEqual(audit['optimizer_steps'],len(history)*steps_per_epoch)
                for state in ck['optimizer_state']['state'].values():
                    self.assertEqual(int(state['step']), ck['epoch']*steps_per_epoch)

    def test_initialization_and_transfer(self):
        audits={n:json.loads((self.run_dir/f'{n}_execution.json').read_text()) for n in ['arsenal_mlp','league_mlp','league_to_arsenal_mlp']}
        for n in ['arsenal_mlp','league_mlp']:
            self.assertIsNone(audits[n]['initial_checkpoint'])
        self.assertEqual(audits['arsenal_mlp']['initial_weights_sha256'],audits['league_mlp']['initial_weights_sha256'])
        self.assertNotEqual(audits['league_mlp']['initial_weights_sha256'],audits['arsenal_mlp']['best_weights_sha256'])
        self.assertEqual(audits['league_to_arsenal_mlp']['initial_weights_sha256'],audits['league_mlp']['best_weights_sha256'])
        for n, audit in audits.items():
            ck=torch.load(self.run_dir/f'{n}_best.pt',weights_only=False)
            model=network(39);model.load_state_dict(ck['model_state'])
            self.assertEqual(state_digest(model),audit['best_weights_sha256'])

    def test_frozen_comparison_precedes_specialization(self):
        comparison=json.loads((self.run_dir/'frozen_validation_comparison.json').read_text())
        self.assertEqual(set(comparison['models']), {'arsenal_mlp','league_mlp'})
        for name, record in comparison['models'].items():
            from experiments.train import digest
            self.assertEqual(record['checkpoint_sha256'],digest(self.run_dir/f'{name}_best.pt'))
            self.assertEqual(record['validation']['n'],74)
        epochs=json.loads((self.run_dir/'league_to_arsenal_mlp_execution.json').read_text())
        self.assertIn('league_mlp_best.pt',epochs['initial_checkpoint'])
