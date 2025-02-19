from src.SignalDetection import SignalDetection
import numpy as np

class Experiment:
    def __init__(self):
        #Initializes an empty list to store SDT objects and their corresponding condition labels.
        self.conditions = []
        
    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        #Adds a SignalDetection object and an optional label to the experiment.
        self.conditions.append((sdt_obj,label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        #Returns sorted false alarm rates and hit rates for plotting the ROC curve
        false_alarm = []
        hit = []
        for condition in self.conditions:
            false_alarm.append[condition.false_alarm_rate()]
            hit.append[condition.hit_rate()]

        

    def compute_auc(self) -> float:

    def plot_roc_curve(self, show_plot: bool = True)