import numpy as np
from sklearn.utils import check_random_state


def make_moons(n_samples=100, 
        xy_ratio=1.0, x_gap=0.0, y_gap=0.0, noise=None):

    """Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    xy_ratio : float, optional (default=1.0)
        ratio of y range over x range. It should be positive
    x_gap : float, optional (default=0.0)
        Gap between y-axis center of two moons. 
        It should be larger than -0.3
    y_gap : float, optional (default=0.0)
        Gap between x-axis center of two moons. 
        It should be larger than -0.3
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    References
    ----------
    .. [1] scikit-learn sklearn.dataset.samples_generator.make_moons
    """

    assert xy_ratio > 0
    assert -0.3 <= x_gap
    assert -0.3 <= y_gap

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(None)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) - x_gap
    outer_circ_y = xy_ratio * np.sin(np.linspace(0, np.pi, n_samples_out)) + y_gap
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in)) + x_gap
    inner_circ_y = xy_ratio * (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - (.5 + y_gap))

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        noise = generator.normal(scale=noise, size=X.shape)
        noise[:,1] = noise[:,1] * xy_ratio
        X += noise

    return X, y

def make_spiral(n_samples_per_class=100, n_classes=2,
        n_rotations=3, gap_between_spiral=0.0, 
        gap_between_start_point=0.0, equal_interval=True,                
        noise=None):

    """Parameters
    ----------
    n_samples_per_class : int, optional (default=100)
        The number of points of a class.
    n_classes : int, optional (default=2)
        The number of spiral
    n_rotations : int, optional (default=3)
        The rotation number of spiral
    gap_between_spiral : float, optional (default=0.0)
        The gap between two parallel lines
    gap_betweein_start_point : float, optional (default=0.0)
        The gap between spiral origin points
    equal_interval : Boolean, optional (default=True)
        Equal interval on a spiral line if True.
        Else their intervals are proportional to their radius.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    color : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.
    """

    assert 1 <= n_classes and type(n_classes) == int

    generator = check_random_state(None)

    X_array = []
    theta = 2 * np.pi * np.linspace(0, 1, n_classes + 1)[:n_classes]

    for c in range(n_classes):

        t_shift = theta[c]
        x_shift = gap_betweein_start_point * np.cos(t_shift)
        y_shift = gap_betweein_start_point * np.sin(t_shift)

        if equal_interval:
            t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class) ** (1/2))
        else:
            t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class))

        x = (1 + gap_between_spiral) * t * np.cos(t + t_shift) + x_shift
        y = (1 + gap_between_spiral) * t * np.sin(t + t_shift) + y_shift

        X = np.concatenate((x, y))

        if noise is not None:
            X += generator.normal(scale=noise, size=X.shape)
        
        X = X.T
        X_array.append(X)

    X = np.concatenate(X_array)
    color = np.asarray([c for c in range(n_classes) for _ in range(n_samples_per_class)])

    return X, color

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