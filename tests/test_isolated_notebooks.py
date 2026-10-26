"""Check that all active notebooks and exported verdicts use the corrected study."""
import json
import unittest
import nbformat
import numpy as np
import pandas as pd
from experiments.isolated_arsenal import ROOT, OUTPUT


class IsolatedNotebooks(unittest.TestCase):
    def test_executed_sequence_and_graphics(self):
        review = ROOT / 'results/arsenal_isolated_review'
        records = json.loads((review/'notebook_execution.json').read_text())
        self.assertEqual(len(records), 5)
        self.assertEqual([r['notebook'][:2] for r in records[:4]], ['01','02','03','04'])
        for record in records:
            n = nbformat.read(ROOT/'notebooks'/record['notebook'], as_version=4)
            nbformat.validate(n)
            cells = [c for c in n.cells if c.cell_type=='code']
            self.assertEqual(len(cells), record['code_cells'])
            self.assertTrue(all(c.execution_count is not None for c in cells))
            self.assertFalse(any(o.output_type=='error' for c in cells for o in c.outputs))
            source = '\n'.join(c.source for c in cells)
            self.assertNotIn('study.fit_stage', source)
            self.assertNotIn('study.begin', source)
            self.assertNotIn("results/test_review", source)
            self.assertIn('from experiments.isolated_arsenal import OUTPUT', source)
        self.assertEqual(len(list((review/'figures').glob('*.png'))),14)

    def test_all_verdicts_match_corrected_results(self):
        folder = ROOT/'results/arsenal_isolated_review/test_review'
        metrics = pd.read_csv(OUTPUT/'evaluation/metrics.csv').query("split=='test'")
        for row in metrics.itertuples():
            table = pd.read_csv(folder/f'{row.model}_arsenal_test_review.csv')
            saved = pd.read_csv(OUTPUT/'evaluation'/f'{row.model}_test_predictions.csv')
            self.assertEqual(table.match_id.tolist(),saved.match_id.tolist())
            self.assertEqual(len(table),75)
            home = table.home_team.eq('Arsenal')
            np.testing.assert_allclose(table.p_win,np.where(home,saved.p_H,saved.p_A),atol=1e-7)
            self.assertAlmostEqual(table.correct.mean(),row.accuracy)
            self.assertTrue(table.home_goals.notna().all())
