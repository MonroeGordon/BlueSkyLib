import cupy as cp
import numpy as np
from numpy.linalg._linalg import EighResult

def eigen(matrix: np.ndarray | cp.ndarray,
          uplo: str="L",
          device: str="cpu") -> EighResult | tuple[cp.ndarray]:
    '''
    Calculate eigenvalues and eigenvectors of a square matrix.
    :param matrix: Square matrix.
    :param uplo: Specifies whether the lower or upper triangular of the matrix is used.
    :param device: CPU or GPU device.
    :return: A tuple containing eigenvalues and eigenvectors.
    '''
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("@ eigen: parameter 'matrix' must be a square matrix (2-dimensional ndarray).")

    if device == "cpu":
        nmtx = matrix

        if isinstance(nmtx, cp.ndarray):
            nmtx = cp.asnumpy(nmtx)

        return np.linalg.eigh(nmtx, "L" if uplo.lower() == "l" else "U")
    else:
        cmtx = matrix

        if isinstance(cmtx, np.ndarray):
            cmtx = cp.asarray(cmtx)

        return cp.linalg.eigh(cmtx, "L" if uplo.lower() == "l" else "U")