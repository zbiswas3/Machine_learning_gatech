import numpy as np
import itertools
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import matplotlib.cm as cm
import copy
import logging
from sklearn.decomposition import PCA
from collections import Counter
from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns

def plot_cluster(clf, X, name, n_components, cv_type):
    Y_ = clf.predict(X)
    fig, ax2 = plt.subplots()
    fig.set_size_inches(10, 8)
    for i, (mean, cov) in enumerate(zip(clf.means_, clf.covariances_)):
        color = cm.nipy_spectral(Y_[Y_==i]/ n_components)
        if not np.any(Y_ == i):
            continue
        ax2.scatter(X[Y_ == i, 0], X[Y_ == i, 1], marker=".", s=30, lw=0, alpha=0.7, c=color, edgecolor="k")

        # Plot an ellipse to show the Gaussian component
        try:
            v, w = linalg.eigh(cov)
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            ax2.add_artist(ell)
        except:
            pass

    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"GMM Cluster: {cv_type} model, "
        f"{n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)

    # ----------------------------------------------------------------------------------------

    plt.savefig("hdfs/{}_scaled_clusters_n_comp_{}_{}.png".format(name, n_components, cv_type))
    plt.close()

def plot_gmm(X,y,name):
    # Generate random sample, two components
    np.random.seed(0)
    lowest_bic = np.infty
    bic = []
    n_components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            plot_cluster(gmm, X, name, n_components, cv_type)
            bic.append(gmm.bic(X))
            print("{} BIC for {} components {} model: {}".format(name, n_components, cv_type, bic[-1]))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        try:
            v, w = linalg.eigh(cov)
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        except:
            pass

    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {best_gmm.covariance_type} model, "
        f"{best_gmm.n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    plt.savefig("hdfs/{}_scaled_gmm_final_best_{}_{}.png".format(name, best_gmm.n_components, cv_type))
    plt.close()

if __name__ == '__main__':
    for name in ['Adult_2D', 'DryBeansData_2D']:        
        dataset = pd.read_csv('hdfs/%s.csv' % name)
        y = dataset["target"].to_numpy()
        X = dataset.drop("target", axis=1).to_numpy()
        X = StandardScaler().fit_transform(X)
        plot_gmm(X,y,name)

