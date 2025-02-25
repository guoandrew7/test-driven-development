import unittest
from src.SignalDetection import SignalDetection
from src.Experiment import Experiment

class TestExperiment(unittest.TestCase):
    def test_add_condition(self):
        exp = Experiment()
        sdt = SignalDetection(10, 5, 3, 7)
        exp.add_condition(sdt, "Condition A")
        self.assertEqual(len(exp.conditions), 1)
        self.assertEqual(exp.conditions[0][1], "Condition A")

    def test_sorted_roc_points(self):
        exp = Experiment()
        sdt1 = SignalDetection(40, 10, 20, 30)  # F = 0.4, H = 0.8
        sdt2 = SignalDetection(30, 20, 10, 40)  # F = 0.2, H = 0.6
        exp.add_condition(sdt1, "Cond 1")
        exp.add_condition(sdt2, "Cond 2")
        fa_rates, h_rates = exp.sorted_roc_points()
        self.assertEqual(fa_rates, sorted(fa_rates))  # Ensure sorted by false alarm rate
        self.assertEqual(len(fa_rates), 2)

    def test_compute_auc(self):
        exp = Experiment()
        sdt1 = SignalDetection(0, 0, 0, 1)  # (0,0)
        sdt2 = SignalDetection(1, 0, 1, 0)  # (1,1)
        exp.add_condition(sdt1)
        exp.add_condition(sdt2)
        self.assertAlmostEqual(exp.compute_auc(), 0.5, places=2)

    def test_compute_auc_perfect(self):
        exp = Experiment()
        sdt1 = SignalDetection(0, 0, 0, 1)  # (0,0)
        sdt2 = SignalDetection(1, 0, 0, 1)  # (0,1)
        sdt3 = SignalDetection(1, 0, 1, 0)  # (1,1)
        exp.add_condition(sdt1)
        exp.add_condition(sdt2)
        exp.add_condition(sdt3)
        self.assertAlmostEqual(exp.compute_auc(), 1.0, places=2)

    def test_empty_experiment_raises_value_error(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.sorted_roc_points()

    def test_empty_experiment_auc_raises_value_error(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.compute_auc()

    def test_add_multiple_conditions(self):
        exp = Experiment()
        for i in range(5):
            exp.add_condition(SignalDetection(10 + i, 5, 3, 7), f"Condition {i}")
        self.assertEqual(len(exp.conditions), 5)
        self.assertEqual(exp.conditions[4][1], "Condition 4")

    def test_sorted_roc_points_correct_order(self):
        exp = Experiment()
        sdt1 = SignalDetection(30, 20, 10, 40)  # F = 0.2, H = 0.6
        sdt2 = SignalDetection(40, 10, 20, 30)  # F = 0.4, H = 0.8
        exp.add_condition(sdt2, "Cond 2")
        exp.add_condition(sdt1, "Cond 1")
        fa_rates, h_rates = exp.sorted_roc_points()
        self.assertEqual(fa_rates, [0.2, 0.4])
        self.assertEqual(h_rates, [0.6, 0.8])

if __name__ == "__main__":
    unittest.main()
