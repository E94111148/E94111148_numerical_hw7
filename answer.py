import numpy as np

#  matriax A & vetor b
A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# 初始猜測
x0 = np.zeros_like(b)

# Jacobi 
def jacobi(A, b, x0, tol=1e-10, max_iterations=1000):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    x = x0.copy()
    for i in range(max_iterations):
        x_new = np.dot(D_inv, b - np.dot(R, x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i + 1
        x = x_new
    return x, max_iterations

# Gauss-Seidel 
def gauss_seidel(A, b, x0, tol=1e-10, max_iterations=1000):
    x = x0.copy()
    n = len(b)
    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iterations

# SOR 
def sor(A, b, x0, omega=1.25, tol=1e-10, max_iterations=1000):
    x = x0.copy()
    n = len(b)
    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iterations

# Conjugate Gradient 
def conjugate_gradient(A, b, x0, tol=1e-10, max_iter=1000):
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for k in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, k + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter

# 解法
x_jacobi, iter_jacobi = jacobi(A, b, x0)
x_gs, iter_gs = gauss_seidel(A, b, x0)
x_sor, iter_sor = sor(A, b, x0, omega=1.25)
x_cg, iter_cg = conjugate_gradient(A, b, x0)

# 直接法參考答案
x_exact = np.linalg.solve(A, b)

# 顯示結果
np.set_printoptions(precision=6, suppress=True)

print("*(a)Jacobi")
print("Solution:", np.round(x_jacobi, 6))
print("Iterations:", iter_jacobi)
print()

print("*(b)Gauss-Seidel")
print("Solution:", np.round(x_gs, 6))
print("Iterations:", iter_gs)
print()

print("*(c)SOR Method (ω = 1.25)")
print("Solution:", np.round(x_sor, 6))
print("Iterations:", iter_sor)
print()

print("*(d)Conjugate Gradient")
print("Solution:", np.round(x_cg, 6))
print("Iterations:", iter_cg)
print()

print("Exact solution to compare:")
print("Solution:", np.round(x_exact, 6))
