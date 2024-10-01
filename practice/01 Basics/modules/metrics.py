import numpy as np
import math


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    ed_dist = 0
    for i in range(len(ts1)):
        ed_dist += (ts1[i] - ts2[i])**2
    return np.sqrt(ed_dist)

def simple_average(ts):
  sum = 0
  for val in ts:
    sum += val
  return sum/len(ts)


def deviation_ts(ts):
  pow_average = 0
  sum = 0
  for val in ts:
    sum += val*val
  pow_average = sum/len(ts)
  return math.sqrt(pow_average - math.pow(simple_average(ts), 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0
    norm_ed_dist = math.sqrt(abs(2*len(ts1)*(1-(np.dot(ts1, ts2)-len(ts1)*simple_average(ts1)*simple_average(ts2))/(len(ts1)*deviation_ts(ts1)*deviation_ts(ts2)))))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    dtw_dist = 0

    n = len(ts1)
    m = len(ts2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[:, 0] = float('inf')
    dtw_matrix[0, :] = float('inf')
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
      for j in range(1, m + 1):
        cost = (ts1[i - 1] - ts2[j - 1])*(ts1[i - 1] - ts2[j - 1])
        dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist
