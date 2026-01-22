from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

## Settings
ALPHA = 0.05
METRICS = ["BAC", "F1", "Time"]
METRICS_PLOTS = ["Balanced Accuracy [1]", "F1 score [1]", "Time per test batch [min]"]

## Import metrics
datablocks = np.load("metrics.npy")

## Average metrics
datablocks = np.mean(datablocks, axis=2)

for idx_metrics in range(datablocks.shape[2]):
    print()
    print("Metrik: ", METRICS[idx_metrics])

    metrics = datablocks[:,:,idx_metrics]
    experience_range = np.arange(metrics.shape[0]) + 1

    fig, ax = plt.subplots()
    ax.scatter(experience_range, metrics[:, 0], c='blue', label='DT')
    ax.scatter(experience_range, metrics[:, 1], c='green', label='CNN')
    ax.scatter(experience_range, metrics[:, 2], c='red', label='LSTM')
    ax.set_xlabel("Number of experiment")
    ax.set_ylabel(METRICS_PLOTS[idx_metrics])
    ax.legend()
    fig.savefig(f"Performance_{METRICS[idx_metrics]}.png")

    ## Perform Friedman test
    _, p_value_friedman = friedmanchisquare(metrics[:,0], metrics[:,1], metrics[:,2])
    if p_value_friedman < ALPHA:
        print(METRICS[idx_metrics], ": Friedman-Test: Signifikant difference: ", p_value_friedman)
    else:
        print(METRICS[idx_metrics], ": Friedman-Test: No Signifikant difference: Skip this metric")
        continue

    ## Perform pairwise wilcoxon
    pairwise_compare = np.zeros(shape=(3))
    pairwise_compare[0] = wilcoxon(metrics[:,1], metrics[:,2])[1]
    pairwise_compare[1] = wilcoxon(metrics[:,0], metrics[:,2])[1]
    pairwise_compare[2] = wilcoxon(metrics[:,0], metrics[:,1])[1]
    print("Pairwise Wilcoxon: ", pairwise_compare)

    ## Bonferroni-Holm
    _, corrected_p, _, _ = multipletests(pairwise_compare, method='holm')
    print("Corrected: ", corrected_p)


    ## Calculate effectiveness - Cohen
    def cohen(g1: np.ndarray, g2: np.ndarray):
        if g1.shape != g2.shape:
            raise Exception("Groups of different size.")
        
        g1_mean = np.mean(g1)
        g2_mean = np.mean(g2)
        g1_std = np.std(g1)
        g2_std = np.std(g2)

        s_pooled = np.sqrt((np.square(g1_std) + np.square(g2_std)) / 2)
        d = (g1_mean - g2_mean) / s_pooled
        r = d / np.sqrt(np.square(d) + 4)
        return r
    
    def eta_square(g1: np.ndarray, g2: np.ndarray):
        ## Source: https://ilias.uni-giessen.de/data/JLUG/lm_data/lm_746625/lm3/chapter2/subchapter3/page-2-3-c/index.html

        if g1.shape != g2.shape:
            raise Exception("Groups of different size.")
        
        ## Step 1
        g1_mean = np.mean(g1)
        g2_mean = np.mean(g2)
        mean = (g1_mean + g2_mean) / 2

        ## Step 2
        g1_ss_in = np.sum(np.square(g1 - g1_mean))
        g2_ss_in = np.sum(np.square(g2 - g2_mean))
        ss_in = g1_ss_in + g2_ss_in

        ## Step 3
        g1_ss_out = g1.shape[0] * np.square(g1_mean - mean)
        g2_ss_out = g2.shape[0] * np.square(g2_mean - mean)
        ss_out = g1_ss_out + g2_ss_out

        ## Step 4
        eta_2 = ss_out / (ss_out + ss_in)
        return eta_2

    effectiveness = np.zeros(shape=(3))
    effectiveness[0] = eta_square(metrics[:,1], metrics[:,2])
    effectiveness[1] = eta_square(metrics[:,0], metrics[:,2])
    effectiveness[2] = eta_square(metrics[:,0], metrics[:,1])
    print("Effect Size: ", effectiveness)