#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--n_components", default=2, type=int)
    args = parse.parse_args()
    return args


def main():
    args = get_args()

    # data
    iris = load_iris()
    x = iris.data
    y = iris.target

    # num class
    classify = np.unique(y)
    l = len(classify)

    # X_{c}, shape=(mc, n)
    x_c_list = [x[y == c] for c in classify]

    # \mu_{c}, List[shape=(n,), ...]
    mu_c_list = [np.mean(x, axis=0) for x in x_c_list]

    # \mu, shape=(n,)
    mu = np.mean(x, axis=0)

    # S_{w}, shape=(n, n)
    sw = np.sum([np.dot((x_c_list[i] - mu_c_list[i]).T, (x_c_list[i] - mu_c_list[i])) for i in range(l)], axis=0)

    # S_{b}, shape=(n, n)
    sb = np.sum([len(mu_c_list[i]) * np.expand_dims(mu_c_list[i] - mu, axis=1) * np.expand_dims(mu_c_list[i] - mu, axis=0) for i in range(l)], axis=0)

    # A = S_{w}^{-1} S_{b}
    a = np.array(np.dot(np.mat(sw).I, sb))

    # eigen
    eig_value, eig_vector = np.linalg.eig(a)
    eig_val_index = np.argsort(eig_value)
    eig_val_index = eig_val_index[-1: -(args.n_components + 1): -1]
    max_eig_vector = eig_vector[:, eig_val_index]

    # transform
    result = np.dot(x, max_eig_vector)

    # show
    plt.figure(2)
    plt.scatter(result[:, 0], result[:, 1], marker='o', c=y)
    plt.show()
    return


if __name__ == '__main__':
    main()
