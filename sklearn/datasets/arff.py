"""
ARFF file loader that returns Bunch objects
"""

# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# Copyright 2011 University of Amsterdam
# License: 3-clause BSD style

from .base import Bunch

import numpy as np
import scipy
from scipy.io.arff import loadarff as sp_loadarff
from scipy.sparse import *  # lil_matrix


# Work around bug in MetaData.types in old SciPy:
# types would not have the same order as names()
if scipy.version.version.split('.') < '0.9.0'.split():
    def _types(m):
        return [m._attributes[x][0] for x in m.names()]
else:
    def _types(m):
        return m.types()


def load_arff(f, sparse_output=False):
    """Load an ARFF-formatted file (as used by Weka).

    The last attribute declared in the ARFF file's header will be considered
    the target attribute.

    This function wraps scipy.io.arff.loadarff and has pretty much the same
    limitations.

    The full ARFF format is described at http://weka.wikispaces.com/ARFF.

    Parameters
    ----------
    f : {string, file-like}
        Path name of the data file to load. When used with SciPy >=0.10,
        a file-like is also supported.

    Returns
    -------
    data : Bunch
        A dict-like object holding
        "data": sample vectors
        "DESCR": the dataset's name/description
        "target": numeric classification labels (indices into the following)
        "target_names": symbolic names of classification labels
    """

    data, meta = sp_loadarff(f)

    dtype = float if set(["numeric", "real"]) & set(meta.types()) else int
    empty = lil_matrix if sparse_output else np.empty
    X = empty((data.shape[0], len(data[0]) - 1), dtype=dtype)

    nominal_features = set(i for i, t in enumerate(_types(meta))
                             if t == "nominal")
    feature_values = dict()
    for i, n in enumerate(meta):
        t, values = meta[n]
        if t == 'nominal':
            for v in values:
                feature_values[i] = dict()
                for j, v in enumerate(values):
                    feature_values[i][v] = j

    for j, n in enumerate(meta.names()[:-1]):
        # FIXME we should do a one-hot transformation for nominal features
        if j in nominal_features:
            column = [feature_values[j][v] for v in data[n]]
        else:
            column = data[n]
        if sparse_output:
            for i in xrange(len(column)):
                X[i, j] = column[i]
        else:
            X[:, j] = column

    target_attr = len(meta.names()) - 1
    target_names = sorted(feature_values[target_attr].keys(),
                          key=lambda i: feature_values[target_attr][i])
    y = np.empty(data.shape[0])
    for i, row in enumerate(data):
        y[i] = feature_values[target_attr][row[target_attr]]

    return Bunch(data=X,
                 DESCR=meta.name,
                 target=y,
                 target_names=target_names)
