from io import BytesIO
import numpy as np
from xgboost.core import Booster
from xgboost import XGBModel
import  matplotlib.pyplot as plt

# 自定义绘图工具
def plot_importance_v1(booster, model_name="xgb", columns_name=None, ax=None, height=0.2,
                       xlim=None, ylim=None, title='Feature importance',
                       xlabel='F score', ylabel='Features',
                       importance_type='weight',max_num_features=None,
                       grid=True, show_values=True, **kwargs):
    """Plot importance based on fitted trees.

    Parameters
    ----------
    booster : Booster, XGBModel or dict
        Booster or XGBModel instance, or dict taken by Booster.get_fscore()
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    grid : bool, Turn the axes grids on or off.  Default is True (On).
    importance_type : str, default "weight"
        How the importance is calculated: either "weight", "gain", or "cover"

        * "weight" is the number of times a feature appears in a tree
        * "gain" is the average gain of splits which use the feature
        * "cover" is the average coverage of splits which use the feature
          where coverage is defined as the number of samples affected by the split
    max_num_features : int, default None
        Maximum number of top features displayed on plot. If None, all features will be displayed.
    height : float, default 0.2
        Bar height, passed to ax.barh()
    xlim : tuple, default None
        Tuple passed to axes.xlim()
    ylim : tuple, default None
        Tuple passed to axes.ylim()
    title : str, default "Feature importance"
        Axes title. To disable, pass None.
    xlabel : str, default "F score"
        X axis title label. To disable, pass None.
    ylabel : str, default "Features"
        Y axis title label. To disable, pass None.
    show_values : bool, default True
        Show values on plot. To disable, pass False.
    kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
    """
    assert model_name in ["xgb", "cb", "ext"]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib to plot importance')

    if model_name =="xgb":
        if isinstance(booster, XGBModel):
            importance = booster.get_booster().get_score(
                importance_type=importance_type)
        elif isinstance(booster, Booster):
            importance = booster.get_score(importance_type=importance_type)
        elif isinstance(booster, dict):
            importance = booster
        else:
            raise ValueError('tree must be Booster, XGBModel or dict instance, or array')
        if not importance:
            raise ValueError('Booster.get_score() results in empty')

        if columns_name:
            # print([columns_name[int(k[1:])] for k in importance])
            # print([importance[k] for k in importance])
            tuples = zip([columns_name[int(k[1:])] for k in importance], [importance[k] for k in importance])
            print(tuples)
        else:

            tuples = [(k, importance[k]) for k in importance]
            print(tuples)
    elif model_name =="cb":  # xgboost
        feat_ = booster.get_feature_importance().tolist()
        tuples = zip(columns_name, feat_)
    else:  # ExtraTree   # 极限树
        feat_ = booster.feature_importances_.tolist()
        tuples = zip(columns_name, feat_)
    # tuples =[("name1",5),("name2",4),("name3",7),("name4",2),("name5",9)]
    if max_num_features is not None:
        # pylint: disable=invalid-unary-operand-type
        tuples = sorted(tuples, key=lambda x: x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda x: x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    # ylocs  y 的key,  也就是对应的f0 到F-N的标签
    # values 值为对应的权重值
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


# fig, ax = plt.subplots(figsize=(15, 15))
# plot_importance_v1(ax=ax)
# plt.show()
# print("ok")