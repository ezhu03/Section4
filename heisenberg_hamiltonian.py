import numpy as np
from scipy.sparse import coo_matrix
import time
from matplotlib import pyplot as plt

def build_heisenberg_xxx(N, J=1.0):
    """
    Build the Heisenberg XXX Hamiltonian matrix for N sites with periodic
    boundary conditions, returning a sparse (2^N x 2^N) matrix in CSR format.

    H = J * sum_{i=0}^{N-1} [ S^z_i S^z_{i+1} + 0.5 * ( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} ) ],
    with site indices taken modulo N.
    """
    dim = 2**N

    # Precompute S^z for each site in each basis state
    # s_z_state[s, i] = +1/2 if spin i is up in basis state s, else -1/2
    s_z_state = np.zeros((dim, N), dtype=np.float64)
    for s in range(dim):
        for i in range(N):
            if ((s >> i) & 1) == 1:  # bit i set => spin up
                s_z_state[s, i] = +0.5
            else:
                s_z_state[s, i] = -0.5

    # Helper function: flip bit i in state s
    def flip_spin(state, i):
        return state ^ (1 << i)

    # Lists to accumulate the nonzero entries in COO format
    row_indices = []
    col_indices = []
    values = []

    # Build the Hamiltonian
    for s in range(dim):
        # ----- Diagonal part: sum over neighbors i, (i+1) -----
        diag_val = 0.0
        for i in range(N):
            j = (i + 1) % N
            diag_val += s_z_state[s, i] * s_z_state[s, j]

        if diag_val != 0.0:
            row_indices.append(s)
            col_indices.append(s)
            values.append(diag_val)

        # ----- Off-diagonal (flip-flop) -----
        for i in range(N):
            j = (i + 1) % N

            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1

            # S^+_i S^-_j: site i down (0), site j up (1)
            if bit_i == 0 and bit_j == 1:
                s_prime = flip_spin(s, i)
                s_prime = flip_spin(s_prime, j)
                row_indices.append(s_prime)
                col_indices.append(s)
                values.append(0.5)  # coefficient

            # S^-_i S^+_j: site i up (1), site j down (0)
            if bit_i == 1 and bit_j == 0:
                s_prime = flip_spin(s, i)
                s_prime = flip_spin(s_prime, j)
                row_indices.append(s_prime)
                col_indices.append(s)
                values.append(0.5)  # coefficient

    # Build a COO matrix then convert to CSR
    H_coo = coo_matrix(
        (values, (row_indices, col_indices)),
         shape=(dim, dim),
         dtype=np.float64
    )
    
    # Multiply by J if desired
    H_coo.data *= J
    
    # Finally convert to CSR for more efficient linear algebra
    H_csr = H_coo.tocsr()
    return H_csr

# Example usage

N = 3
J = 1.0
H = build_heisenberg_xxx(N, J)
print("Dimension:", H.shape)
print("Sparsity:", H.nnz)
# For small N, convert to dense for inspection
print(H.toarray())


print('Time complexity of the Heisenberg Hamiltonian is O(N*2^N)')
# --- Timing study for various N ---
# (Using small N only, since the Hilbert space size is 2^N.)
Ns = range(5,17)
times = []
fit = []
start = 0
for n in Ns:
    start = time.perf_counter()
    _ = build_heisenberg_xxx(n, J)
    end = time.perf_counter()
    dt = end - start
    times.append(dt)
    if n == 5:
        fit.append(dt)
        save = dt
    else:
        fit.append(save*(n/5)*(2**(n-5)))
    print(f"N = {n:2d}, Hilbert space dimension = {2**n:5d}, construction time = {dt:.6f} sec")
    
    # Plot the construction time versus N.
plt.figure(figsize=(8, 6))
plt.plot(Ns, times, 'o-', label='Hamiltonian construction time')
plt.plot(Ns, fit, 'r--', label='Fit O(N*2^N)')
plt.xlabel("Number of Spins (N)")
plt.ylabel("Time (seconds)")
plt.title("Dense Hamiltonian Construction Time vs. Chain Length N")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("heisenberg_timing.png")

# ========================================
# QR Algorithm for Diagonalization
# ========================================

def qr_eigen(A, max_iter=1000, tol=1e-8):
    """
    Compute eigenvalues, eigenvectors, and the (nearly) diagonalized matrix using the QR algorithm.
    
    Note: For matrices with repeated eigenvalues, the result may only be block diagonal.
    """
    n = A.shape[0]
    Ak = A.copy()
    Q_total = np.eye(n)
    
    for iteration in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak_new = R @ Q
        Q_total = Q_total @ Q
        
        # Check convergence based on the Frobenius norm of the change.
        if np.linalg.norm(Ak_new - Ak, ord='fro') < tol:
            print(f"Converged in {iteration+1} iterations.")
            Ak = Ak_new
            break
        Ak = Ak_new

    eigenvalues = np.diag(Ak)
    eigenvectors = Q_total
    diagonalized_matrix = Ak
    return eigenvalues, eigenvectors, diagonalized_matrix
def qr_diagonalize_dp(H, iteration=0, max_iterations=1000, tolerance=1e-12, dp=None):
    """
    Recursively performs the basic QR iteration to (approximately) diagonalize H,
    using a top-down dynamic programming approach.

    Parameters
    ----------
    H : numpy.ndarray
        The matrix to diagonalize (assumed square).
    iteration : int
        Current iteration index.
    max_iterations : int
        Maximum allowed number of QR iterations.
    tolerance : float
        Convergence tolerance on off-diagonal norm.
    dp : dict or None
        A dictionary for caching intermediate results:
          dp[k] = the matrix at iteration k.
        If None, a new dictionary is created at the start.

    Returns
    -------
    H_diag : numpy.ndarray
        The matrix after convergence (or reaching the max iterations),
        which should be close to diagonal.
    """
    # If this is the first call, initialize the dictionary with the starting matrix
    if dp is None:
        dp = {}
        # Convert H to float to avoid in-place modifications of the original
        dp[0] = H.astype(float).copy()
    
    # Retrieve the matrix for the current iteration
    H_current = dp[iteration]

    # ----- Stopping condition if we've reached max_iterations -----
    if iteration >= max_iterations:
        return H_current

    # ----- Perform a QR decomposition on H_current -----
    Q, R = np.linalg.qr(H_current)
    H_next = R @ Q  # The next iterate

    # ----- Check convergence by looking at off-diagonal norm -----
    # We measure how large the off-diagonal elements are.
    off_diag = H_next - np.diag(np.diag(H_next))
    off_diag_norm = np.linalg.norm(off_diag, ord='fro')
    if off_diag_norm < tolerance:
        return H_next  # Converged enough

    # ----- Store this next result in dp -----
    dp[iteration + 1] = H_next

    # ----- Recursive call for the next iteration -----
    return qr_diagonalize_dp(H_next, iteration + 1, max_iterations, tolerance, dp=dp)

# --- Example: Diagonalizing the Hamiltonian for N=3 ---
# (For the QR algorithm, we work with a dense matrix.)
H=build_heisenberg_xxx(2, J)
H_dense = H.toarray()

# Perform the QR diagonalization (recursive DP)
H_diagonal_approx = qr_diagonalize_dp(H_dense, max_iterations=200, tolerance=1e-12)
threshold = 0.01
H_diagonal_approx[np.abs(H_diagonal_approx) < threshold] = 0.0
print("\nMatrix after QR algorithm:\n", H_diagonal_approx)
print("Off-diagonal norm:", np.linalg.norm(H_diagonal_approx - np.diag(np.diag(H_diagonal_approx))))

# Compare with numpy's built-in eigenvalues
w, v = np.linalg.eigh(H_dense)  # exact diagonalization of a symmetric/hermitian matrix
print("\nEigenvalues (np.linalg.eigh):\n", w)
print("\nDiagonal from QR result:\n", np.diag(H_diagonal_approx))

from scipy.linalg import lu_factor, lu_solve

def greens_function_lu(H, omega):
    """
    Compute G(omega) = (omega I - H)^(-1) using LU factorization.
    Returns the full NxN inverse as a NumPy array.

    Parameters
    ----------
    H : (N,N) ndarray (real symmetric Hermitian)
    omega : float

    Returns
    -------
    G : (N,N) ndarray, the Green's function at frequency omega.
    """
    N = H.shape[0]
    A = omega * np.eye(N) - H
    # Factor A = L U (with partial pivot)
    lu, piv = lu_factor(A)
    
    # We want the inverse of A => solve A X = I for X
    # Do this column-by-column
    I = np.eye(N)
    G = np.zeros((N, N), dtype=A.dtype)
    for col_idx in range(N):
        e_col = I[:, col_idx]   # standard basis vector
        x_col = lu_solve((lu, piv), e_col)
        G[:, col_idx] = x_col
    return G

def greens_function_cholesky(H, omega):
    """
    Compute G(omega) = (omega I - H)^(-1) using Cholesky factorization,
    assuming (omega I - H) is positive definite.

    Parameters
    ----------
    H : (N,N) real-symmetric (Hermitian)
    omega : float

    Returns
    -------
    G : (N,N) ndarray, Green's function at frequency omega.
    """
    N = H.shape[0]
    A = omega * np.eye(N) - H
    
    # Cholesky factorization A = L L^T  (requires A to be positive definite)
    L = np.linalg.cholesky(A)
    
    # Solve for inverse by columns: A X = I => X = A^-1
    I = np.eye(N)
    G = np.zeros((N, N), dtype=A.dtype)
    for col_idx in range(N):
        # Step 1: Solve L * y = e_col
        e_col = I[:, col_idx]
        y = np.linalg.solve(L, e_col)
        # Step 2: Solve L^T * x = y
        x = np.linalg.solve(L.T, y)
        G[:, col_idx] = x
    
    return G
def main_greens_function(N=16):
    H = build_heisenberg_xxx(N, J=1.0).toarray()
        # 2) Get eigenvalues to identify spectral range
    evals, evecs = np.linalg.eigh(H)
    lam_min = np.min(evals)
    lam_max = np.max(evals)
    print(f"H has eigenvalues in [{lam_min:.3f}, {lam_max:.3f}]")
    print(sorted([round(val, 2) for val in evals]))

        # 3) Choose frequency range
    w_vals = np.linspace(lam_min - 0.5, lam_max + 0.5, 300)  # 300 points

    traces = []
    for w in w_vals:
        A = w * np.eye(2**N) - H
            # We skip singularities if the matrix is near singular
        if np.linalg.cond(A) > 1e12: 
            # skip or set some large placeholder
            traces.append(np.nan)
            continue
        print(w)
        G_w = greens_function_lu(H, w)
            # sum of diagonal
        trace_val = np.trace(G_w)
        traces.append(trace_val)

        # 4) Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(w_vals, traces, marker='o', markersize=2, linestyle='-', label=r'$ \mathrm{Tr}[G(\omega)]$')
    plt.axvline(lam_min, color='gray', linestyle='--', label='Min eigenvalue')
    plt.axvline(lam_max, color='gray', linestyle='--', label='Max eigenvalue')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\mathrm{Tr}[G(\omega)]$')
    plt.title(f'Green\'s function trace vs. frequency (N={N})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("trace_vs_frequency.png")

#main_greens_function(N=10)

def build_single_magnon_matrix(N, J=1.0):
    """
    Produces a single-magnon subspace Hamiltonian whose energies
    match E(p) = 2J sin^2(p/2), from 0 up to 2J.
    """
    M = np.zeros((N, N), dtype=float)
    for n in range(N):
        def Sz(i):
            # Down spin at site n => -1/2, all others +1/2
            return -0.5 if (i % N) == n else 0.5

        diag_val = (Sz(n-1)*Sz(n) + Sz(n)*Sz(n+1))  
        # Usually diag_val = -0.5 if the chain is "Heisenberg XXX."
        # We put a minus sign in front so it becomes +0.5, etc.
        M[n, n] = -diag_val * J

        # Flip-flop signs also negated:
        M[(n+1) % N, n] -= 0.5 * J
        M[(n-1) % N, n] -= 0.5 * J
    M += 0.5 * np.eye(N)

    return M
def energy_of_magnon(M, p):
    """
    Given the NxN matrix M (representing H in the single spin-down subspace),
    compute E(p) = <p|M|p> / <p|p>,
    where |p>_n = e^{i p n}, n=0..N-1.
    M must be NxN. p is a float.

    Returns the real number E(p).
    """
    N = M.shape[0]
    # Construct the vector v_p of shape (N,)
    # v_p(n) = e^{ i p n }, but we can keep it as complex128
    n_array = np.arange(N)
    v_p = np.exp(1j * p * n_array)

    # <p|p> = sum_n |v_p(n)|^2 = N
    norm_p = np.sqrt(np.vdot(v_p, v_p))  # Should be sqrt(N)
    # For clarity, let's keep the vector normalized:
    v_p = v_p / norm_p

    # Compute w = M|p>
    w = M @ v_p  # matrix-vector product (N,)

    # Then E = <p|w>
    E = np.vdot(v_p, w)  # a complex number in general, but should be real
    return E.real
import matplotlib.pyplot as plt

def main_magnon_dispersion(N=30, J=1.0):
    # 1) Build the single-magnon Hamiltonian matrix of size NxN
    M = build_single_magnon_matrix(N, J=J)
    
    # 2) For each k, define p = 2 pi k / N,
    #    compute E_num(k), and compare with E_theory(k) = 2J sin^2(p/2).
    ks = np.arange(N)
    p_vals = 2.0 * np.pi * ks / N
    
    E_numeric = []
    E_theory = []
    
    for p in p_vals:
        E_num = energy_of_magnon(M, p)
        E_numeric.append(E_num)
        
        E_th = 2.0 * J * (np.sin(p/2.0))**2
        E_theory.append(E_th)
    
    E_numeric = np.array(E_numeric)
    E_theory = np.array(E_theory)
    
    # 3) Plot
    plt.figure(figsize=(7,4.5))
    plt.plot(ks, E_numeric, 'bo-', label='E_numeric')
    plt.plot(ks, E_theory, 'r--', label=r'$2J\,\sin^2\left(\frac{p}{2}\right)$')
    plt.title(f"Single-Magnon Energy vs. k, for N={N}")
    plt.xlabel(r"$k$ (where $p = 2\pi k/N$)")
    plt.ylabel(r"Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("magnon_dispersion.png")

main_magnon_dispersion(N=30, J=1.0)
