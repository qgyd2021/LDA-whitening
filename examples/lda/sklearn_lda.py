#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--n_components", default=2, type=int)
    args = parse.parse_args()
    return args


def main():
    """
    sklearn LDA
    http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    """
    iris = load_iris()
    x = iris.data
    y = iris.target

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x, y)
    x_new = lda.transform(x)

    plt.figure(2)
    plt.scatter(x_new[:, 0], x_new[:, 1], marker='o', c=y)
    plt.show()
    return


if __name__ == '__main__':
    main()
