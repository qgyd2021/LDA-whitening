#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import numpy as np

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int
    )
    parser.add_argument(
        "--x_standard_deviation",
        default=3,
        type=int
    )
    parser.add_argument(
        "--y_standard_deviation",
        default=1,
        type=int
    )
    parser.add_argument("--rotation_angle", default=45, type=int)

    parser.add_argument("--fig_size", default=10, type=int)
    parser.add_argument("--w1_length", default=4, type=int)
    parser.add_argument("--w2_length", default=4, type=int)

    parser.add_argument(
        "--output_file",
        default=(project_path / "docs/pictures/pca.jpg").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    x = np.random.randn(args.num_samples) * args.x_standard_deviation
    y = np.random.randn(args.num_samples) * args.y_standard_deviation
    samples = np.array([x, y])

    theta = args.rotation_angle / 180 * np.pi
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    samples_ = np.dot(rotation_matrix, samples)

    plt.figure(figsize=(args.fig_size, args.fig_size))
    plt.scatter(x=samples_[0], y=samples_[1], marker="d", color="red", s=2)

    w1 = np.array([1, 0])
    w1_ = np.dot(rotation_matrix, w1) * args.w1_length
    # plt.arrow(0, 0, w1_[0], w1_[1], color="black", head_width=0.3, head_length=0.5)
    # https://blog.csdn.net/qq_36387683/article/details/101377416
    plt.annotate(
        "w1", xy=(0, 0), xytext=(w1_[0], w1_[1]),
        color="black", weight="normal", fontsize="xx-large",
        arrowprops=dict(
            lw=3,
            arrowstyle="<-",
        )
    )

    w2 = np.array([0, 1])
    w2_ = np.dot(rotation_matrix, w2) * args.w2_length
    # plt.arrow(0, 0, w2_[0], w2_[1], color="black", head_width=0.3, head_length=0.5)
    plt.annotate(
        "w2", xy=(0, 0), xytext=(w2_[0], w2_[1]),
        color="black", weight="normal", fontsize="xx-large",
        arrowprops=dict(
            lw=3,
            arrowstyle="<-",
        )
    )

    plt.axis("equal")
    plt.savefig(args.output_file)
    plt.show()
    return


if __name__ == '__main__':
    main()
