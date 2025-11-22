from cupyx.scipy.linalg import lu as clu
from linalg.eigen import eigen
from scipy.linalg import lu as slu

import cupy as cp
import numpy as np

def eigdecomp(matrix: np.ndarray | cp.ndarray,
               device: str="cpu") -> tuple:
    '''
    Perform eigendecomposition on a matrix.
    :param matrix: A square matrix (2-dimensional ndarray).
    :param device: CPU or GPU device.
    :return: Eigenvalues and eigenvectors.
    '''
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("@ eigdecomp: parameter 'matrix' must be a square matrix (2-dimensional ndarray).")

    if device == "cpu":
        nmtx = matrix

        if isinstance(nmtx, cp.ndarray):
            nmtx = cp.asnumpy(nmtx)

        e_vals, e_vecs = eigen(nmtx)

        sorted_indices = np.argsort(e_vals)[::-1]
    else:
        cmtx = matrix

        if isinstance(cmtx, np.ndarray):
            cmtx = cp.asarray(cmtx)

        e_vals, e_vecs = eigen(cmtx)

        sorted_indices = np.argsort(e_vals)[::-1]

    return e_vals[sorted_indices], e_vecs[:, sorted_indices]

def lu_decomp(matrix: np.ndarray | cp.ndarray,
              permute_l: bool=False,
              overwrite_matrix: bool=False,
              check_finite: bool=False,
              device: str="cpu") -> tuple:
    '''
    Compute the LU decomposition of a square matrix.
    :param matrix: A square matrix (2-dimensional ndarray).
    :param permute_l: If set, permutes L by multiplying it with P
    :param overwrite_matrix: If set, overwrites matrix.
    :param check_finite: If set, checks that all matrix values are finite.
    :param device: CPU or GPU device.
    :return: If not permuting, Permuted (P), Lower (L), and upper (U) triangular matrices, otherwise Permuted Lower
    (PL), and upper (U) matrices.
    '''
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("@ lu: parameter 'matrix' must be a square matrix (2-dimensional ndarray).")

    if device == "cpu":
        nmtx = matrix

        if isinstance(nmtx, cp.ndarray):
            nmtx = cp.asnumpy(nmtx)

        return slu(nmtx, permute_l, overwrite_matrix, check_finite)
    else:
        cmtx = matrix

        if isinstance(cmtx, np.ndarray):
            cmtx = cp.asarray(cmtx)

        return clu(cmtx, permute_l, overwrite_matrix, check_finite)

def svd_decomp(matrix: np.ndarray | cp.ndarray,
               device: str="cpu") -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
    '''
    Perform Singular Value Decomposition (SVD) on a matrix.
    :param matrix: Input matrix (2-dimensional ndarray).
    :param device: CPU or GPU device.
    :return: U, S, V matrices.
    '''
    if device == "cpu":
        u, s, v = np.linalg.svd(matrix)
    else:
        u, s, v = cp.linalg.svd(matrix)

    return u, s, v