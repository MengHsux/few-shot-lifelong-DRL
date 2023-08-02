#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st


###############################################################################
### re-scale the bottom part of the y-axis
def bottom_scale(arr, scale=[-200, -60, -50], dim=None):
    if scale is None:
        return arr
    if dim is None:
        arr = np.array(arr).reshape(-1);
        arr_n = np.zeros(arr.shape)
        ratio = (scale[2] - scale[1]) / (scale[2] - scale[0])

        def shrink(num):
            if num < scale[2]:
                return scale[2] - ratio * (scale[2] - num)
            else:
                return num

        for idx, ii in enumerate(arr):
            arr_n[idx] = shrink(ii)
        return arr_n
    else:
        arr_n = []
        for row in arr:
            arr_n.append(bottom_scale(row, scale=scale, dim=None))
        return np.array(arr_n)


p_output = 'output/navi_v1'


###############################################################################
def plot_performance():
    ic = range(0, 50);
    num = 200;
    mark = 10

    rews_sllrl_0 = np.load(os.path.join(p_output, 'rewards_sllrl.npy'))[ic, :num]

    def ave_rews_stats(arr):
        ave_rews = np.mean(arr, axis=0)
        ave_ave_rews = np.mean(ave_rews)
        err_ave_rews = np.std(ave_rews, ddof=1) / np.sqrt(len(ave_rews))
        return ave_ave_rews, err_ave_rews

    stats = np.zeros((5, 2))
    stats[0] = ave_rews_stats(rews_sllrl_0)

    print('Our Method, mean: %.3f, standard error: %.3f' % (stats[0, 0], stats[0, 1]))

    def conf_int(arr):
        arr_stats = np.zeros((3, arr.shape[1]))
        for idx in range(arr.shape[1]):
            col = arr[:, idx]
            arr_stats[0, idx] = np.mean(col)
            down, up = st.t.interval(0.95, len(col) - 1, loc=np.mean(col),
                                     scale=st.sem(col))
            arr_stats[1, idx], arr_stats[2, idx] = down, up
        return arr_stats

    rews_sllrl = conf_int(rews_sllrl_0)

    scale = [-200, -60, -50]
    rews_sllrl = bottom_scale(rews_sllrl, scale=scale, dim=1)

    plt.figure(figsize=(4, 3), dpi=200)
    alpha = 0.1;
    ms = 4;
    lw = 1;
    mew = 1
    tick_size = 8;
    label_size = 10
    x = np.arange(1, num + 1)

    plt.fill_between(x, rews_sllrl[1], rews_sllrl[2],
                     color='red', alpha=alpha)
    plt.plot(x, rews_sllrl[0], color='red', lw=lw,
             marker='x', markevery=mark, ms=ms, mew=mew, mfc='white')

    plt.xlabel('Learning Episodes', fontsize=label_size)
    plt.ylabel('Return', fontsize=label_size)
    plt.xticks(np.arange(0, num + 1, num // 5), fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.grid(axis='y', ls='-', lw=0.2)
    plt.grid(axis='x', ls='-', lw=0.2)

    plt.axis([0, num, -60, 2])
    yticks = [-200, -50, -40, -30, -20, -10, 0]
    plt.yticks(np.arange(-60, 10, 10), yticks)

    plt.show()


plot_performance()
