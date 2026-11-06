"""Check the real training/evaluation workflow on its fixed data split."""
import unittest
from examples.demo import train_experiment


class TrainingTests(unittest.TestCase):
    def test_held_out_performance_and_split(self):
        report, model, labels, predictions = train_experiment()
        self.assertEqual(report['train_samples'] + report['test_samples'], 1797)
        self.assertEqual(len(labels), 360)
        self.assertEqual(len(predictions), len(labels))
        self.assertGreater(report['test_accuracy'], 0.85)
        self.assertGreater(report['test_macro_f1'], 0.85)
        self.assertGreater(report['test_accuracy'], report['dummy_test_accuracy'] + 0.5)
        self.assertEqual(model.named_steps['standardscaler'].n_samples_seen_, report['train_samples'])


if __name__ == '__main__':
    unittest.main()
