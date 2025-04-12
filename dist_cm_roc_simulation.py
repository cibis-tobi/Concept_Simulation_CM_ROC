import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
from scipy.stats import gaussian_kde

# Setup
np.random.seed(42)
n_samples = 500
x = np.linspace(0, 1, 1000)

# Initial parameters
INIT_THRESH = 0#0.5
INIT_POS_MEAN = 0.7
INIT_NEG_MEAN = 0.3
STD_DEV = 0.1

# Global state
y_true = None
y_prob = None
y_prob_pos = None
y_prob_neg = None
roc_points = []  # Stores (FPR, TPR) for selected thresholds

# Matplotlib figure with 3 subplots
fig, (ax_dist, ax_cm, ax_roc) = plt.subplots(1, 3, figsize=(18, 5))
plt.subplots_adjust(bottom=0.45)

# Slider axes
ax_thresh_slider = plt.axes([0.2, 0.3, 0.6, 0.03])
ax_pos_slider = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_neg_slider = plt.axes([0.2, 0.1, 0.6, 0.03])

# Sliders
threshold_slider = Slider(ax_thresh_slider, "Decision Threshold", 0.0, 1.0, valinit=INIT_THRESH, valstep=0.01)
pos_mean_slider = Slider(ax_pos_slider, "Positive Class Mean", 0.1, 0.9, valinit=INIT_POS_MEAN, valstep=0.01)
neg_mean_slider = Slider(ax_neg_slider, "Negative Class Mean", 0.1, 0.9, valinit=INIT_NEG_MEAN, valstep=0.01)

def generate_data(pos_mean, neg_mean):
    global y_true, y_prob, y_prob_pos, y_prob_neg, roc_points

    # Reset saved ROC points
    roc_points = []

    # Generate synthetic data
    y_true_pos = np.ones(n_samples // 2)
    y_true_neg = np.zeros(n_samples // 2)

    y_prob_pos = np.random.normal(loc=pos_mean, scale=STD_DEV, size=n_samples // 2)
    y_prob_neg = np.random.normal(loc=neg_mean, scale=STD_DEV, size=n_samples // 2)

    y_prob_pos = np.clip(y_prob_pos, 0, 1)
    y_prob_neg = np.clip(y_prob_neg, 0, 1)

    y_true = np.concatenate([y_true_neg, y_true_pos])
    y_prob = np.concatenate([y_prob_neg, y_prob_pos])

def update_plots(threshold, update_roc=True):
    ax_dist.clear()
    ax_cm.clear()
    ax_roc.clear()

    # KDE for distributions
    kde_pos = gaussian_kde(y_prob_pos, bw_method=0.15)
    kde_neg = gaussian_kde(y_prob_neg, bw_method=0.15)

    pos_pdf = kde_pos(x)
    neg_pdf = kde_neg(x)

    # Distribution plot
    ax_dist.plot(x, neg_pdf, label="Negative Class", color='blue')
    ax_dist.plot(x, pos_pdf, label="Positive Class", color='red')
    ax_dist.axvline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold:.2f}")

    ax_dist.fill_between(x, 0, pos_pdf, where=(x >= threshold), color='green', alpha=0.3, label='TP')
    ax_dist.fill_between(x, 0, neg_pdf, where=(x >= threshold), color='orange', alpha=0.3, label='FP')
    ax_dist.fill_between(x, 0, pos_pdf, where=(x < threshold), color='purple', alpha=0.3, label='FN')
    ax_dist.fill_between(x, 0, neg_pdf, where=(x < threshold), color='cyan', alpha=0.3, label='TN')

    ax_dist.set_title("Prediction Distributions")
    ax_dist.set_xlabel("Predicted Probability")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax_dist.grid(True)

    # Confusion matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',cbar=False,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    # ROC point update
    if update_roc:
        TP = cm[1, 1]
        FN = cm[1, 0]
        FP = cm[0, 1]
        TN = cm[0, 0]
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        roc_points.append((FPR, TPR))

    # ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    if roc_points:
        fprs, tprs = zip(*sorted(roc_points))
        ax_roc.plot(fprs, tprs, marker='o', color='darkred', label="Selected Thresholds")
        ax_roc.scatter(fprs[-1], tprs[-1], color='black', zorder=5)

    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (based on selected thresholds)")
    ax_roc.legend()
    ax_roc.grid(True)

    fig.canvas.draw_idle()

# Initial data and plot
generate_data(INIT_POS_MEAN, INIT_NEG_MEAN)
update_plots(INIT_THRESH)

# Event handlers
def on_threshold_change(val):
    update_plots(val, update_roc=True)

def on_class_mean_change(val):
    # Reset data and ROC
    pos_mean = pos_mean_slider.val
    neg_mean = neg_mean_slider.val
    generate_data(pos_mean, neg_mean)

    # Reset threshold slider to default
    threshold_slider.set_val(INIT_THRESH)

    # Plot with reset
    update_plots(INIT_THRESH, update_roc=True)

# Connect sliders
threshold_slider.on_changed(on_threshold_change)
pos_mean_slider.on_changed(on_class_mean_change)
neg_mean_slider.on_changed(on_class_mean_change)

plt.show()
