from collections import defaultdict
from dataclasses import dataclass
import os
import yaml
import time


import numpy as np
import scipy.io
import scipy.linalg

from src.common import NDArrayFloat
from src.linalg import get_numpy_eigenvalues


@dataclass
class Performance:
    time: float = 0.0
    relative_error: float = 0.0

def givens_matrix(vec: NDArrayFloat):
    x = vec.copy()
    G = np.eye(2)
    a = x[0]
    c = x[1]
    if(a != 0 and c!= 0):
        G[0][0] = a/np.sqrt(a**2+c**2)
        G[1][1] = a/np.sqrt(a**2+c**2)
        G[0][1] = c/np.sqrt(a**2+c**2)
        G[1][0] = -c/np.sqrt(a**2+c**2)
    return(G)

counter = 0

def is_small(A: NDArrayFloat) -> bool:
    shape = np.shape(A)
    if (shape == (2, 2) or shape == (1, 1) or shape == ()):
        return True
    
def eigen_of_small(A: NDArrayFloat) -> NDArrayFloat:
    eigv = []
    if (np.shape(A) == (2, 2)):
        for x in np.roots([1, -np.trace(A), np.linalg.det(A)]):
            eigv.append(x)
        return eigv
    if (np.shape(A) == (1, 1)):
        eigv.append(A[0][0])
        return eigv
    if (np.shape(A) == ()):
        return eigv

def qr(M: NDArrayFloat):
    global counter
    A = M.copy()
    eigv = []
    if (is_small(A)):
        for x in eigen_of_small(A):
            eigv.append(x)
        return(eigv)
    n = len(A[0])
    c = 0
    while (counter < 2000):
        if (is_small(A)):
            for x in eigen_of_small(A):
                eigv.append(x)
            return(eigv)
        if (counter == 0):
            r_angle = n
            for i in range(n-1, 0, -1):
                if (np.abs(A[i][i-1]) <= 10e-8*(np.abs(A[i-1][i-1]) + np.abs(A[i][i]))):
                    A[i][i-1] = 0
                    for x in qr(A[i:r_angle,i:r_angle]):
                        eigv.append(x)
                    r_angle = i
            A = A[0:r_angle,0:r_angle]
            if (is_small(A)):
                for x in eigen_of_small(A):
                    eigv.append(x)
                return(eigv)
            counter += 1
        c += 1
        n = len(A[0])
        x = np.zeros(3)
        if (A[1][0] != 0):
            x[0] = ((A[0][0]-A[n-2][n-2])*(A[0][0]-A[n-1][n-1])-A[n-1][n-2]*A[n-2][n-1])/A[1][0] + A[0][1]
        x[1] = A[0][0]+A[1][1]-A[n-2][n-2]-A[n-1][n-1]
        x[2] = A[2][1]
        if (c == 11 or c == 21):
            ks = 1.5 * (np.abs(A[n-1][n-2]) + np.abs(A[n-2][n-3]))
            kss = (np.abs(A[n-1][n-2]) + np.abs(A[n-2][n-3]))**2
            x[0] = A[0][0]**2 - A[0][0]*(ks) + kss + A[0][1]*A[1][0]
            x[1] = A[1][0]*(A[0][0] + A[2][2] - ks)
            x[2] = A[1][0]*A[2][1]
        if (c >= 31):
            for x in np.diag(A):
                eigv.append(x)
            return eigv
        sign = np.sign(x[0])
        if (sign == 0):
            sign = 1
        norm = np.linalg.norm(x)*sign
        if (norm == 0):
            norm = 1
        h = x[0] + norm
        if (h == 0):
            h = 1
        gamma = (h)/norm
        u = np.zeros(3)
        u[0] = 1
        u[1:3] = x[1:3] / h
        v = gamma * u
        A[:3, 0:] = A[:3, 0:] - np.outer(v, u @ A[:3, 0:])
        A_t = A[0:,:3].T
        A_t = A_t - np.outer(v, u@A_t)
        A[0:,:3] = A_t.T
        for i in range(n-3):
            sign = 1
            if (A[i+1][i] < 0):
                sign = -1
            norm = np.linalg.norm(A[i+1:i+4,i])*sign
            h = A[i+1][i] + norm
            if (norm == 0):
                norm = 1
            if (h == 0):
                h = 1
            gamma = (h)/norm
            u = np.zeros(3)
            u[0] = 1
            u[1:3] = A[i+2:i+4,i] / h
            v = gamma * u
            A[i+1:i+4, 0:] = A[i+1:i+4, 0:] - np.outer(v, u @ A[i+1:i+4,0:])
            A_t = A[0:,i+1:i+4].T
            A_t = A_t - np.outer(v, u@A_t)
            A[0:,i+1:i+4] = A_t.T
        Q = givens_matrix(A[n-2:n, n-3])
        A[n-2:n, n-3:n] = Q @ A[n-2:n, n-3:n]
        A[0:, n-2:n] = A[0:, n-2:n] @ Q.T
        r_angle = n
        for i in range(n-1, 0, -1):
            if (np.abs(A[i][i-1]) <= 10e-8*(np.abs(A[i-1][i-1]) + np.abs(A[i][i]))):
                A[i][i-1] = 0
                for x in qr(A[i:r_angle,i:r_angle]):
                    eigv.append(x)
                r_angle = i
                c = 0
                
        A = A[0:r_angle,0:r_angle]
        counter += 1
    return eigv

def hessenberg_reduction(A: NDArrayFloat) -> NDArrayFloat:
    n = len(A[0])
    A_hess = A.copy()
    Q = np.eye(n)
    for i in range(n-2):
        x = A_hess[i + 1 : , i].copy()
        sign = np.sign(x[0])
        if (sign == 0):
            sign = 1
        norm = np.linalg.norm(x)*sign
        if (norm == 0):
            continue
        h = x[0] + norm
        gamma = (h)/norm
        u = np.zeros_like(x)
        u[0] = 1
        u[1:] = x[1:] / h
        v = gamma * u
        A_hess[i+1:, i:] = A_hess[i+1:, i:] - np.outer(v, u @ A_hess[i+1:, i:])
        A_t = A_hess[0:,i+1:].T
        A_t = A_t - np.outer(v, u@A_t)
        A_hess[0:,i+1:] = A_t.T

    return A_hess

def get_all_eigenvalues(A: NDArrayFloat) -> NDArrayFloat:
    A_hess = hessenberg_reduction(A)
    eigen = qr(A_hess)
    eig = np.zeros(len(A[0]), dtype=complex)
    for x in range(len(eigen)):
        eig[x] = np.complex64(eigen[x])
    global counter
    counter = 0
    return eig

def run_test_cases(
    path_to_homework: str, path_to_matrices: str
) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open(os.path.join(path_to_homework, "matrices.yaml"), "r") as f:
        matrix_filenames = yaml.safe_load(f)
    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")
        A = scipy.io.mmread(os.path.join(path_to_matrices, matrix_filename)).todense().A
        perf = performance_by_matrix[matrix_filename]
        t1 = time.time()
        eigvals = get_all_eigenvalues(A)
        t2 = time.time()
        eigvals1 = eigvals.copy()
        perf.time += t2 - t1
        eigvals_exact = get_numpy_eigenvalues(A)
        eigvals_exact.sort()
        eigvals1.sort()
        perf.relative_error = np.median(
            np.abs(eigvals_exact - eigvals1) / np.abs(eigvals_exact)
        )
    return performance_by_matrix


if __name__ == "__main__":
    path_to_homework = os.path.join("practicum_7\homework\\advanced")
    path_to_matrices = os.path.join("practicum_6\homework\\advanced\matrices")
    performance_by_matrix = run_test_cases(
        path_to_homework=path_to_homework,
        path_to_matrices=path_to_matrices,
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )