import numpy as np
from sklearn.utils import check_random_state


def make_swiss_roll(n_samples=100, n_rotations=1.5, 
        gap=0, thickness=0.0, width=10.0):

    """Generate a swiss roll dataset.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples in swiss roll
    n_rotations : float, optional (default=1.5)
        The number of turns
    gap : float, optional (default=1.0)
        The gap between two roll planes
    thickness : float, optional (default=0.0)
        The thickness of roll plane
    
    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.
    Returns
    -------
    X : array of shape [n_samples, 3]
        The points.
    color : array of shape [n_samples]
        The normalized univariate position of the sample according to the main 
        dimension of the points in the manifold. Its scale is bounded in [0, 1]

    References
    ----------
    .. [1] scikit-learn sklearn.dataset.samples_generator.make_swiss_roll
    """
    generator = check_random_state(None)

    t = n_rotations * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = (1 + gap) * t * np.cos(t)
    y = width * (generator.rand(1, n_samples) - 0.5)
    z = (1 + gap) * t * np.sin(t)

    X = np.concatenate((x, y, z))
    X += thickness * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    color = (t - t.min()) / (t.max() - t.min())

    return X, color