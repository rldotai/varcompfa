"""
Radial basis function features.
"""
import numpy as np
from .feature_base import Feature


class RBF(Feature):
    """Radial basis function feature. Uses a 'gaussian' activation.

    Notes
    -----
    Scale factor determines the "width" of each RBF.
    Suggested scale factor proportional to (range**2) for each dimension,
    where `range` refers to the maximum value less the minimum.
    Use a smaller value for tighter RBFs, larger values for wider ones with
    more generalization.
    """
    def __init__(self, points, scale=1.0):
        self.points = np.atleast_2d(points)
        nf, nd = points.shape # num_features, num_dimensions

        # Accomodate scalar or vector scaling factors
        if np.ndim(scale) == 0:
            scale = scale*np.ones((nf, nd))
        elif np.ndim(scale) == 1:
            if len(scale) == nf:
                scale = np.repeat(scale, nd).reshape(nf, nd)
            else:
                scale = np.tile(scale, nf).reshape(nf, nd)
        # Store other relevant parameters
        self.sigma = 1/scale
        self.num_features = nf
        self.num_dimensions = nd

    def __call__(self, x):
        x = np.atleast_2d(x)
        y = np.array([i - self.points for i in x])
        z = -(y**2)
        tot = np.einsum('ijk,jk->ij', z, self.sigma)
        act = np.exp(tot)
        return np.squeeze(act).T

    def __len__(self):
        return self.num_features


class NRBF(Feature):
    """Normalized radial basis function

    See: http://www.cs.colostate.edu/~anderson/res/rl/matt-icnn97.pdf

    Notes
    -----

    Scale factor determines the "width" of each RBF.
    Suggested scale factor proportional to (range**2) for each dimension,
    where `range` refers to the maximum value less the minimum.
    Use a smaller value for tighter RBFs, larger values for wider ones with
    more generalization.
    Typically, for normalized RBFs, you want less generalization.

    Empirically, it seems to play better with larger learning rates,
    presumably because its maximum value is capped at one, and the norm of the
    feature vector is always one.
    """
    def __init__(self, points, scale=1.0):
        self.points = np.atleast_2d(points)
        nf, nd = points.shape # num_features, num_dimensions

        # Accomodate scalar or vector scaling factors
        if np.ndim(scale) == 0:
            scale = scale*np.ones((nf, nd))
        elif np.ndim(scale) == 1:
            if len(scale) == nf:
                scale = np.repeat(scale, nd).reshape(nf, nd)
            else:
                scale = np.tile(scale, nf).reshape(nf, nd)
        # Store other relevant parameters
        self.sigma = 1/scale
        self.num_features = nf
        self.num_dimensions = nd

    def __call__(self, x):
        x = np.atleast_2d(x)
        y = np.array([i - self.points for i in x])
        z = -(y**2)
        tot = np.einsum('ijk,jk->ij', z, self.sigma)
        act = np.exp(tot)

        # Normalization
        act = np.einsum('ij,i->ij', act, 1/np.sum(act, axis=1))
        return np.squeeze(act).T

    def __len__(self):
        return self.num_features


# TODO: Find an elegant way of implementing this, maybe from `binary_vector.py`
# class RTU(vcf.features.Feature):
#     """Radial threshold unit"""
#     def __init__(self, points, scale=1.0, num_active=1):
#         self.points = np.atleast_2d(points)
#         nf, nd = points.shape # num_features, num_dimensions

#         # Accomodate scalar or vector scaling factors
#         if np.ndim(scale) == 0:
#             scale = scale*np.ones((nf, nd))
#         elif np.ndim(scale) == 1:
#             if len(scale) == nf:
#                 scale = np.repeat(scale, nd).reshape(nf, nd)
#             else:
#                 scale = np.tile(scale, nf).reshape(nf, nd)
#         # Store other relevant parameters
#         self.sigma = 1/scale
#         self.num_features = nf
#         self.num_dimensions = nd

#         def someones(*ixs):
#             """A vector of zeros except at the given indices"""
#             ret = np.zeros(nf)
#             ret[[ixs]] = 1
#             return ret

#         self.someones = someones

#     def __call__(self, x):
#         x = np.atleast_2d(x)
#         y = np.array([i - self.points for i in x])
#         z = -(y**2)
#         tot = np.einsum('ijk,jk->ij', z, self.sigma)
#         act = np.exp(tot)

#         # Get highest activated values
#         high = np.argsort(act, axis=0)[:, :self.num_active]


#         return np.squeeze(act).T

#     def __len__(self):
#         return self.num_features
