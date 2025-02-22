from src.SignalDetection import SignalDetection
from src.Experiment import Experiment
import unittest
import numpy as np
from scipy.stats import norm

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.exp = Experiment()

    def test_add_condition(self):
        sdt = SignalDetection(10, 5, 5, 10)
        self.exp.add_condition(sdt, label="Test Condition")
        self.assertEqual(len(self.exp.conditions), 1)

    def test_sorted_roc_points(self):
        sdt1 = SignalDetection(30, 10, 10, 50) 
        sdt2 = SignalDetection(40, 10, 20, 30)  
        self.exp.add_condition(sdt2, "Cond B")
        self.exp.add_condition(sdt1, "Cond A")
        false_alarm_rates, hit_rates = self.exp.sorted_roc_points()
        self.assertTrue(false_alarm_rates[0] <= false_alarm_rates[1])  

    def test_compute_auc(self):
        self.exp.add_condition(SignalDetection(50, 0, 0, 50), "Perfect")
        self.exp.add_condition(SignalDetection(0, 50, 50, 0), "Worst")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 1.0, places=3)  

    def test_auc_for_linear_roc(self):
        self.exp.add_condition(SignalDetection(0, 50, 0, 50), "Cond1")
        self.exp.add_condition(SignalDetection(50, 0, 50, 0), "Cond2")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=3) 

    def test_empty_experiment_error(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()
        with self.assertRaises(ValueError):
            self.exp.compute_auc()

    def test_edge_case_one_condition(self):
        self.exp.add_condition(SignalDetection(25, 25, 25, 25), "Single Condition")
        false_alarm_rates, hit_rates = self.exp.sorted_roc_points()
        self.assertEqual(len(false_alarm_rates), 1)
        self.assertEqual(len(hit_rates), 1)

    def test_plot_roc_curve(self):
        self.exp.add_condition(SignalDetection(30, 10, 20, 40), "Plot Test")
        try:
            self.exp.plot_roc_curve(show_plot=False)
        except Exception as e:
            self.fail(f"plot_roc_curve() raised {type(e)} unexpectedly!")

if __name__ == "__main__":
    unittest.main()