from linalg.decomposition import lu_decomp

import cupy as cp
import numpy as np

def _backward_substitution(u: np.ndarray | cp.ndarray,
                           y: np.ndarray | cp.ndarray,
                           device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Perform backward substitution to solve Ux = y for x.
    :param u: Upper triangular matrix.
    :param y: Solution vector from forward substitution (1-dimensional ndarray).
    :param device: CPU or GPU device.
    :return: Solution vector x.
    '''
    n = u.shape[0]

    if device == "cpu":
        nu = u
        ny = y

        if isinstance(nu, cp.ndarray):
            nu = cp.asnumpy(nu)

        if isinstance(ny, cp.ndarray):
            ny = cp.asnumpy(ny)

        x = np.zeros_like(ny)

        for i in range(n - 1, -1, -1):
            x[i] = (ny[i] - np.dot(nu[i, i + 1:], x[i + 1:])) / nu[i, i]
    else:
        cu = u
        cy = y

        if isinstance(cu, np.ndarray):
            cu = cp.asarray(cu)

        if isinstance(cy, np.ndarray):
            cy = cp.asarray(cy)

        x = cp.zeros_like(cy)

        for i in range(n - 1, -1, -1):
            x[i] = (cy[i] - cp.dot(cu[i, i + 1:], x[i + 1:])) / cu[i, i]

    return x

def _forward_substitution(l: np.ndarray | cp.ndarray,
                          b: np.ndarray | cp.ndarray,
                          device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Perform forward substitution to solve Ly = b for y.
    :param l: Lower triangular matrix.
    :param b: Right-hand side vector (1-dimensional ndarray).
    :return: Solution vector y.
    '''
    n = l.shape[0]

    if device == "cpu":
        nl = l
        nb = b

        if isinstance(nl, cp.ndarray):
            nl = cp.asnumpy(nl)

        if isinstance(nb, cp.ndarray):
            nb = cp.asnumpy(nb)

        y = np.zeros_like(nb)

        for i in range(n):
            y[i] = (nb[i] - np.dot(nl[i, :i], y[:i])) / nl[i, i]
    else:
        cl = l
        cb = b

        if isinstance(cl, np.ndarray):
            cl = cp.asarray(cl)

        if isinstance(cb, np.ndarray):
            cb = cp.asarray(cb)

        y = cp.zeros_like(cb)

        for i in range(n):
            y[i] = (cb[i] - cp.dot(cl[i, :i], y[:i])) / cl[i, i]

    return y

def gaussian_elimination(a: np.ndarray | cp.ndarray,
                         b: np.ndarray | cp.ndarray,
                         device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Solve the system of linear equations ax = b using Gaussian elimination.
    :param a: Coefficient matrix.
    :param b: Constant vector.
    :param device: CPU or GPU device.
    :return: Solution vector x.
    '''
    if a.ndim != 2:
        raise ValueError("@ gaussian_elimination: parameter 'a' must be a 2-dimensional ndarray.")

    br = b.flatten()

    if device == "cpu":
        na = a
        nb = br

        if isinstance(na, cp.ndarray):
            na = cp.asnumpy(na)

        if isinstance(nb, cp.ndarray):
            nb = cp.asnumpy(nb)

        n = len(nb)
        ab = np.hstack([na, nb.reshape(-1, 1)])

        for i in range(n):
            ab[i] = ab[i] / ab[i][i]

            for j in range(i + 1, n):
                ab[j] = ab[j] - ab[j][i] * ab[i]

        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            x[i] = ab[i][-1] - np.sum(ab[i][i + 1:n] * x[i + 1:n])
    else:
        ca = a
        cb = br

        if isinstance(ca, np.ndarray):
            ca = cp.asarray(ca)

        if isinstance(cb, np.ndarray):
            cb = cp.asarray(cb)

        n = len(cb)
        ab = cp.hstack([ca, cb.reshape(-1, 1)])

        for i in range(n):
            ab[i] = ab[i] / ab[i][i]

            for j in range(i + 1, n):
                ab[j] = ab[j] - ab[j][i] * ab[i]

        x = cp.zeros(n)

        for i in range(n - 1, -1, -1):
            x[i] = ab[i][-1] - cp.sum(ab[i][i + 1:n] * x[i + 1:n])

    return x

def lu_solve(a: np.ndarray | cp.ndarray,
             b: np.ndarray | cp.ndarray,
             device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Solve the linear system ax = b using LU decomposition.
    :param a: Coefficient matrix (2-dimensional ndarray).
    :param b: Right-hand side vector (1-dimensional ndarray).
    :param device: CPU or GPU device.
    :return: Solution vector x.
    '''
    if a.ndim != 2:
        raise ValueError("@ lu_solve: parameter 'a' must be a 2-dimensional ndarray.")

    br = b.flatten()

    l, u = lu_decomp(a, device=device)

    y = _forward_substitution(l, br, device)

    return _backward_substitution(u, y, device)

def matrix_inv(a: np.ndarray | cp.ndarray,
               b: np.ndarray | cp.ndarray,
               device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Solve the system of linear equations ax = b using matrix inversion.
    :param a: Coefficient matrix.
    :param b: Constant vector.
    :param device: CPU or GPU device.
    :return: Solution vector x.
    '''
    if a.ndim != 2:
        raise ValueError("@ matrix_inv: parameter 'a' must be a 2-dimensional ndarray.")

    br = b.flatten()

    if device == "cpu":
        na = a
        nb = br

        if isinstance(na, cp.ndarray):
            na = cp.asnumpy(na)

        if isinstance(nb, cp.ndarray):
            nb = cp.asnumpy(nb)

        a_inv = np.linalg.inv(na)
        x = np.dot(a_inv, nb)
    else:
        ca = a
        cb = br

        if isinstance(ca, np.ndarray):
            ca = cp.asarray(ca)

        if isinstance(cb, np.ndarray):
            cb = cp.asarray(cb)

        a_inv = cp.linalg.inv(ca)
        x = cp.dot(a_inv, cb)

    return x