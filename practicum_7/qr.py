import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat


# # def qr(A: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]: # на выходе Q и R
# #     n = A.shape[0]
# #     print(A)
# #     print("~"*15)
# #     Q = np.zeros_like(A)
# #     R = np.zeros_like(A)
# #     for i in range(n):
# #         v = A[:, i]
# #         for j in range(i):
# #             R[j, i] = np.dot(Q[:, j], A[:, i]) # скалярное умножение
# #             v = v - R[j, i] * Q[:, j] 
# #         R[i, i] = np.linalg.norm(v)
# #         Q[:, i] = v / R[i, i]
# #     return Q, R
    
# def qr_old(A: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]: # на выходе Q и R
#     n = A.shape[0]
#     Q = np.zeros_like(A)
#     R = np.zeros_like(A)
#     W = A.copy()
    
#     for j in range(n):
#         w_j_norm = np.linalg.norm(W[: , j])
#         Q[: , j ] = W[: , j] / w_j_norm # W[: , j] == w_j^j
#         for i in range(j):
#             R[i,j] = A[: , j] @ Q[:, i]
#         a_j_norm = np.linalg.norm(A[:, j])
#         R[j,j] = np.sqrt(a_j_norm**2 - np.sum(R[ :j , j] ** 2))
#         if np.isnan(R[j,j]):
#             R[j,j] = 0
#         for k in range(j+1,n):
#             prod = W[: , k] @ Q[: , j]
#             W[:, k ] = W[: , k] - prod * Q[:, j]
#     return Q,R

# # def hessenberg_reduction(A: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
# #     n = len(A[0])
# #     A_hess = A.copy()
# #     Q = np.eye(n)
# #     for i in range(n-2):
# #         x = A_hess[i+1:, i].copy()
# #         # print(x)
# #         x_norm_signed = np.linalg.norm(x) * np.sign(x[0])
# #         x[0] += x_norm_signed
# #         u = x / (np.linalg.norm(x))
# #         A_hess[i+1:, i:] = A_hess[i+1:, i:] - 2 * np.outer(u, u@A_hess[i+1:, i:])
# #         H = np.eye(len(x)) - 2*(np.outer(u,u)) # внешнее произведение
# #         # A_hess[i+1:, i:] = H @ A_hess[i+1:, i:]
# #         A_t = A_hess[0:,i+1:].T
# #         A_t = A_t - 2 * np.outer(u,u@A_t)
# #         A_hess[0:,i+1:] = A_t.T
# #         # A_hess[0:,i+1:] = A_hess[0:,i+1:]@H
# #         # Q[i+1:, i:] = H @ Q[i+1:, i:]


# #         # H = np.eye(len(x)) - 2*(np.outer(u,u)) # внешнее произведение
# #         # A_hess[i+1:, i:] = H @ A_hess[i+1:, i:]
# #         # A_hess[0:,i+1:] = A_hess[0:,i+1:]@H
# #         # Q[i+1:, i:] = H @ Q[i+1:, i:]

# #     return A_hess

# def hessenberg_reduction(A: NDArrayFloat) -> NDArrayFloat:
#     n = len(A[0])
#     A_hess = A.copy()
#     Q = np.eye(n)
#     for i in range(n-2):
#         x = A_hess[i + 1 : , i].copy()
#         sign = np.sign(x[0])
#         if (sign == 0):
#             sign = 1
#         norm = np.linalg.norm(x)*sign
#         if (norm == 0):
#             norm = 1
#         h = x[0] + norm
#         if (h == 0):
#             h = 1
#         gamma = (h)/norm
#         u = np.zeros_like(x)
#         u[0] = 1
#         u[1:] = x[1:] / h
#         v = gamma * u
#         A_hess[i+1:, i:] = A_hess[i+1:, i:] - np.outer(v, u @ A_hess[i+1:, i:])
#         A_t = A_hess[0:,i+1:].T
#         A_t = A_t - np.outer(v, u@A_t)
#         A_hess[0:,i+1:] = A_t.T

#     return A_hess

# def givens_iter(A: NDArrayFloat):
#     n = len(A[0])
#     m = np.mean(A)
#     step = np.diag(np.full(n, A[n-1][n-1]))
#     A -= step
#     G_matricies = dict()
#     for j in range(n-2):
#         a = A[j][j]
#         c = A[j+1][j]
#         G = np.eye(2)
#         if(a != 0 and c!= 0):
#             G[0][0] = a/np.sqrt(a**2+c**2)
#             G[1][1] = a/np.sqrt(a**2+c**2)
#             G[0][1] = c/np.sqrt(a**2+c**2)
#             G[1][0] = -c/np.sqrt(a**2+c**2)
#         A[j:j+2, :] = G@A[j:j+2, :]
#         G_matricies[j] = G.T
#     for j in range(n-2):
#         A[:, j:j+2] = A[:,j:j+2]@G_matricies[j]
#     A += step
#     return A

# def qr_via_givens_rotation(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
#     n = len(A[0])
#     A_k = A.copy()
#     A_kk = A_k.copy()
#     current_eigen = n - 1
#     for i in range(n_iters):
#         step = np.diag(np.full(n, A[current_eigen][current_eigen]))
#         A_k -= step
#         G_matricies = dict()
#         for j in range(n-2):
#             a = A_k[j][j]
#             c = A_k[j+1][j]
#             G = np.eye(2)
#             if(a != 0 and c!= 0):
#                 G[0][0] = a/np.sqrt(a**2+c**2)
#                 G[1][1] = a/np.sqrt(a**2+c**2)
#                 G[0][1] = c/np.sqrt(a**2+c**2)
#                 G[1][0] = -c/np.sqrt(a**2+c**2)
#             A_k[j:j+2, :] = G@A_k[j:j+2, :]
#             G_matricies[j] = G.T
#         A_kk = A_k
#         for j in range(n-2):
#             A_kk[:, j:j+2] = A_kk[:,j:j+2]@G_matricies[j]
#         A_kk += step
#         A_k = A_kk
#         # print(A_kk)
        
#     # print(A_kk)
#     return np.diag(A_kk)

# def householder_refl (A: NDArrayFloat, i) -> NDArrayFloat:
#     norm = np.linalg.norm(A[0:3,0])*np.sign(A[0][0])
#     h = A[0][0] + norm
#     gamma = (h)/norm
#     u = np.zeros(3)
#     # u[0] = 1
#     # u[1] = A[1][0]/h
#     # u[2] = A[2][0]/h

#     u[0] = 1
#     u[1:3] = A[1:3,0] / h

#     v = gamma * u
#     A = A - np.outer(v, u @ A)

#     return A

# def givens_rotation(A: NDArrayFloat):
#     A_g = A.copy()
#     G = np.eye(2)
#     a = A_g[0][0]
#     c = A_g[1][0]
#     if(a != 0 and c!= 0):
#         G[0][0] = a/np.sqrt(a**2+c**2)
#         G[1][1] = a/np.sqrt(a**2+c**2)
#         G[0][1] = c/np.sqrt(a**2+c**2)
#         G[1][0] = -c/np.sqrt(a**2+c**2)
#     A_g[0:2, :] = G @ A_g[0:2, :]
#     return(A_g)

# def givens_matrix(vec: NDArrayFloat):
#     x = vec.copy()
#     G = np.eye(2)
#     a = x[0]
#     c = x[1]
#     if(a != 0 and c!= 0):
#         G[0][0] = a/np.sqrt(a**2+c**2)
#         G[1][1] = a/np.sqrt(a**2+c**2)
#         G[0][1] = c/np.sqrt(a**2+c**2)
#         G[1][0] = -c/np.sqrt(a**2+c**2)
#     return(G)
# def householder_matrix (vec: NDArrayFloat) -> NDArrayFloat:
#     x = vec.copy()
#     x[0] += np.sign(x[0])*np.linalg.norm(x)
#     u = (x) / (np.linalg.norm(x))
#     H = np.eye(len(x)) - 2*(np.outer(u,u)) 
#     return H

# counter = 1

# def qr(A: NDArrayFloat):
#     global counter
#     # A = M.copy()
#     eigv = []
#     n = len(A[0])
#     c = 0
#     while (counter < 10000):
#         c+= 1
#         print(A)
            
#         n = len(A[0])
#         if (np.shape(A) == (2, 2)):
#             for x in np.linalg.eigvals(A):
#                 eigv.append(x)
#             return eigv
#         if (np.shape(A) == (1, 1)):
#             eigv.append(A[0][0])
#             return eigv
#         if (np.shape(A) == ()):
#             return eigv
#         x = np.zeros(3)
#         x[0] = ((A[0][0]-A[n-2][n-2])*(A[0][0]-A[n-1][n-1])-A[n-1][n-2]*A[n-2][n-1])/A[1][0] + A[0][1]
#         x[1] = A[0][0]+A[1][1]-A[n-2][n-2]-A[n-1][n-1]
#         x[2] = A[2][1]
#         # Q = householder_matrix(x)
#         if (c == 10):
#             print('Hey!')
#             c = 0
#             givens_iter(A)
#             continue
#         sign = np.sign(x[0])
#         if (sign == 0):
#             sign = 1
#         norm = np.linalg.norm(x)*sign
#         if (norm == 0):
#             norm = 1
#         h = x[0] + norm
#         gamma = (h)/norm
#         u = np.zeros(3)
#         u[0] = 1
#         u[1:3] = x[1:3] / h
#         v = gamma * u
#         A[:3, 0:] = A[:3, 0:] - np.outer(v, u @ A[:3, 0:])
#         A_t = A[0:,:3].T
#         A_t = A_t - np.outer(v, u@A_t)
#         A[0:,:3] = A_t.T
#         for i in range(n-3):
#             norm = np.linalg.norm(A[i+1:i+4,i])*np.sign(A[i+1][i])
#             h = A[i+1][i] + norm
#             if (norm == 0):
#                 norm = 1
#             if (h == 0):
#                 h = 1
#             gamma = (h)/norm
#             u = np.zeros(3)
#             u[0] = 1
#             u[1:3] = A[i+2:i+4,i] / h
#             v = gamma * u
#             A[i+1:i+4, 0:] = A[i+1:i+4, 0:] - np.outer(v, u @ A[i+1:i+4,0:])
#             A_t = A[0:,i+1:i+4].T
#             A_t = A_t - np.outer(v, u@A_t)
#             A[0:,i+1:i+4] = A_t.T
#         Q = givens_matrix(A[n-2:n, n-3])
#         A[n-2:n, n-3:n] = Q @ A[n-2:n, n-3:n]
#         A[0:, n-2:n] = A[0:, n-2:n] @ Q.T
#         r_angle = n
#         for i in range(n-1, 0, -1):
#             if (np.abs(A[i][i-1]) < 10e-8*(np.abs(A[i-1][i-1]) + np.abs(A[i][i]))):
#                 A[i][i-1] = 0
#                 for x in qr(A[i:r_angle,i:r_angle]):
#                     eigv.append(x)
#                 r_angle = i
                
#         A = A[0:r_angle,0:r_angle]
#         counter += 1
#         print('\r', counter, end = '')
#     return eigv

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
def householder_matrix (vec: NDArrayFloat) -> NDArrayFloat:
    x = vec.copy()
    u = np.zeros(3)
    x[0] += np.sign(x[0])*np.linalg.norm(x)
    if (np.linalg.norm(x) != 0):
        u = (x) / (np.linalg.norm(x))
    H = np.eye(len(x)) - 2*(np.outer(u,u)) 
    return H

counter = 0

def qr(M: NDArrayFloat):
    global counter
    A = M.copy()
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
    n = len(A[0])
    c = 0
    while (counter < 1500):
        if (np.shape(A) == (2, 2)):
            for x in np.roots([1, -np.trace(A), np.linalg.det(A)]):
                eigv.append(x)
            return eigv
        if (np.shape(A) == (1, 1)):
            eigv.append(A[0][0])
            return eigv
        if (np.shape(A) == ()):
            return eigv
        if (counter == 0):
            r_angle = n
            for i in range(n-1, 0, -1):
                if (np.abs(A[i][i-1]) <= 10e-8*(np.abs(A[i-1][i-1]) + np.abs(A[i][i]))):
                    A[i][i-1] = 0
                    for x in qr(A[i:r_angle,i:r_angle]):
                        eigv.append(x)
                    r_angle = i
            A = A[0:r_angle,0:r_angle]
            if (np.shape(A) == (2, 2)):
                for x in np.roots([1, -np.trace(A), np.linalg.det(A)]):
                    eigv.append(x)
                return eigv
            if (np.shape(A) == (1, 1)):
                eigv.append(A[0][0])
                return eigv
            if (np.shape(A) == ()):
                return eigv
            counter += 1
        c+= 1
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
        if (c == 31):
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


def get_numpy_eigenvalues(A):
    return np.linalg.eigvals(A)

if __name__ == "__main__":
    for i in range (1):
        A = np.array(
            [
                [9.0, 0.0, 5.0, 0.0],
                [10.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, -1.0, -1.0],
            ]
        )
        
        np.random.seed(42)
        A = np.random.rand(10,10)
        eigvals_exact = get_numpy_eigenvalues(A)
        eigvals_exact.sort()

        A = hessenberg_reduction(A)
        print('hess done')
        eig = qr(A)
        eigvals = np.zeros_like(eigvals_exact, dtype=complex)
        for i in range(len(eig)):
            eigvals[i] = np.complex64(eig[i])
        eigvals.sort()
        print(np.power(-2, 1/2, dtype=complex))
        relative_error = np.median(
            np.abs(eigvals_exact - eigvals) / np.abs(eigvals_exact)
        )
        # print(eigvals, eigvals_exact)
        print('\n',relative_error)
        
        


