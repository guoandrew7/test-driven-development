from src.SignalDetection import SignalDetection
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self):
        # Initializes an empty list to store SDT objects and their corresponding condition labels.
        self.conditions = []
        
    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        # Adds a SignalDetection object and an optional label to the experiment.
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        # Returns sorted false alarm rates and hit rates for plotting the ROC curve.

        if not self.conditions:
            raise ValueError("No conditions are present")

        false_alarm_rates = []
        hit_rates = []
        for sdt_obj, _ in self.conditions:  # Unpacking the tuple correctly
            false_alarm_rates.append(sdt_obj.false_alarm_rate())
            hit_rates.append(sdt_obj.hit_rate())
        
        sorted_indices = np.argsort(false_alarm_rates)
        sorted_false_alarm_rates = np.array(false_alarm_rates)[sorted_indices].tolist()
        sorted_hit_rates = np.array(hit_rates)[sorted_indices].tolist()
        
        return sorted_false_alarm_rates, sorted_hit_rates

    def compute_auc(self) -> float:
        # Computes the Area Under the Curve (AUC) for the stored SDT conditions.
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return trapezoid(hit_rates, false_alarm_rates)

    def plot_roc_curve(self, show_plot: bool = True):
        # Plots ROC
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance Level')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()
        if show_plot:
            plt.show()