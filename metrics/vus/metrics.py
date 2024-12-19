from .utils.metrics import metricor
from .analysis.robustness_eval import generate_curve


def get_range_vus_roc(score, labels, slidingWindow):
    grader = metricor()
    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    # R_AUC_ROC = 0
    # R_AUC_PR = 0
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, 2*slidingWindow)
    metrics = {'VUS_ROC': VUS_ROC, 'VUS_PR': VUS_PR}

    return metrics
