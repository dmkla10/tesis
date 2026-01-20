import numpy as np

def fvbm_unnormalized_prob(x, b, W):
    """
    Compute the unnormalized probability (Boltzmann factor) of a configuration x
    for a Fully Visible Boltzmann Machine (FVBM) with x_i ∈ {-1, 1}.
    
    Parameters
    ----------
    x : np.ndarray, shape (p,)
        Binary vector with entries -1 or +1.
    b : np.ndarray, shape (p,)
        Bias parameters.
    W : np.ndarray, shape (p, p)
        Symmetric interaction matrix with zeros on the diagonal.

    Returns
    -------
    float
        Unnormalized probability exp(b^T x + 0.5 * x^T W x).
    """
    x = np.asarray(x)
    b = np.asarray(b)
    W = np.asarray(W)

    # Linear term
    linear = np.dot(b, x)

    # Quadratic term (use x^T W x, but divide by 2 to avoid double counting)
    quadratic = 0.5 * np.dot(x, np.dot(W, x))

    energy = linear + quadratic
    return np.exp(energy)
