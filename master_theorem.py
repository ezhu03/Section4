import random
import time
import matplotlib.pyplot as plt
import numpy as np

def add_matrix(A, B):
    """Elementwise addition of two matrices A and B."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def subtract_matrix(A, B):
    """Elementwise subtraction of matrix B from A."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def split_matrix(M):
    """Splits a given matrix M into 4 quadrants."""
    n = len(M)
    mid = n // 2
    A11 = [row[:mid] for row in M[:mid]]
    A12 = [row[mid:] for row in M[:mid]]
    A21 = [row[:mid] for row in M[mid:]]
    A22 = [row[mid:] for row in M[mid:]]
    return A11, A12, A21, A22

def combine_quadrants(C11, C12, C21, C22):
    """Combines four quadrants into a single matrix."""
    top = [c11_row + c12_row for c11_row, c12_row in zip(C11, C12)]
    bottom = [c21_row + c22_row for c21_row, c22_row in zip(C21, C22)]
    return top + bottom

def divide_and_conquer_mult(A, B):
    """
    Multiply two matrices A and B using a naive divide-and-conquer approach.
    The recursion divides each matrix into 4 n/2 x n/2 submatrices and computes:
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
    """
    n = len(A)
    if n == 1:
        # Base case: 1x1 matrix.
        return [[A[0][0] * B[0][0]]]
    
    # Split matrices into quadrants.
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Recursively compute the 8 products.
    M1 = divide_and_conquer_mult(A11, B11)
    M2 = divide_and_conquer_mult(A12, B21)
    C11 = add_matrix(M1, M2)
    
    M3 = divide_and_conquer_mult(A11, B12)
    M4 = divide_and_conquer_mult(A12, B22)
    C12 = add_matrix(M3, M4)
    
    M5 = divide_and_conquer_mult(A21, B11)
    M6 = divide_and_conquer_mult(A22, B21)
    C21 = add_matrix(M5, M6)
    
    M7 = divide_and_conquer_mult(A21, B12)
    M8 = divide_and_conquer_mult(A22, B22)
    C22 = add_matrix(M7, M8)
    
    # Combine the 4 quadrants into a single matrix.
    return combine_quadrants(C11, C12, C21, C22)

def generate_matrix(n, low=0, high=10):
    """Generates an n x n matrix with random integers between low and high."""
    return [[random.randint(low, high) for _ in range(n)] for _ in range(n)]

n = 4  # Make sure n is a power of 2.
A = generate_matrix(n)
B = generate_matrix(n)
C = divide_and_conquer_mult(A, B)
    
print("Matrix A:")
for row in A:
    print(row)
print("\nMatrix B:")
for row in B:
    print(row)
print("\nProduct C = A * B:")
for row in C:
    print(row)

print("Critical Exponent = log(2, 8) = 3")

def time_divide_and_conquer(n):
    """Generates two random n x n matrices and times the multiplication."""
    A = generate_matrix(n)
    B = generate_matrix(n)
    start_time = time.perf_counter()
    _ = divide_and_conquer_mult(A, B)
    end_time = time.perf_counter()
    return end_time - start_time

# Choose a range of matrix sizes (n must be a power of 2).
sizes = [2**i for i in range(2, 10)]  # n = 4, 8, 16, 32, 64, 128
runtimes = []

for n in sizes:
    t = time_divide_and_conquer(n)
    runtimes.append(t)
    print(f"Size: {n}x{n}, Time: {t:.6f} seconds")

# Plot the results.
plt.figure(figsize=(8, 6))
plt.plot(sizes, runtimes, 'bo-', label="Divide-and-Conquer Runtime")
plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.title("Runtime of Naive Divide-and-Conquer Matrix Multiplication")
plt.grid(True)

# To see if the growth is cubic, we plot a reference curve ~ n^3.
# Normalize the n^3 curve for comparison.
n_cubic = np.array(sizes, dtype=float) ** 3
# Normalize to have the same scale as runtimes.
scale_factor = runtimes[0] / (sizes[0] ** 3)
plt.plot(sizes, scale_factor * n_cubic, 'r--', label=r"$\Theta(n^3)$")

plt.legend()
plt.show()
plt.savefig("master_runtime.png")
# Additionally, estimate the exponent from the measured data using a log-log plot.
log_sizes = np.log(sizes)
log_runtimes = np.log(runtimes)
slope, intercept = np.polyfit(log_sizes, log_runtimes, 1)
print(f"Estimated exponent from runtime data: {slope:.2f}")

def strassen(A, B):
    """
    Multiply two matrices A and B using Strassen’s algorithm.
    """
    n = len(A)
    # Base case: when matrix is 1x1.
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Split the matrices into quadrants.
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Compute the 7 intermediate matrices (recursively).
    M1 = strassen(add_matrix(A11, A22), add_matrix(B11, B22))
    M2 = strassen(add_matrix(A21, A22), B11)
    M3 = strassen(A11, subtract_matrix(B12, B22))
    M4 = strassen(A22, subtract_matrix(B21, B11))
    M5 = strassen(add_matrix(A11, A12), B22)
    M6 = strassen(subtract_matrix(A21, A11), add_matrix(B11, B12))
    M7 = strassen(subtract_matrix(A12, A22), add_matrix(B21, B22))
    
    # Compute the quadrants of the result matrix C.
    C11 = add_matrix(subtract_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(add_matrix(subtract_matrix(M1, M2), M3), M6)
    
    # Combine the quadrants into a single matrix.
    return combine_quadrants(C11, C12, C21, C22)

# Example usage:
C = strassen(A, B)
    
print("Matrix A:")
for row in A:
    print(row)
print("\nMatrix B:")
for row in B:
    print(row)
print("\nProduct C = A * B (via Strassen’s algorithm):")
for row in C:
    print(row)

print("Critical Exponent = log(2, 7) = 2.81")

def time_strassen(n):
    """Generates two random n x n matrices and times Strassen’s multiplication."""
    A = generate_matrix(n)
    B = generate_matrix(n)
    start_time = time.perf_counter()
    _ = strassen(A, B)
    end_time = time.perf_counter()
    return end_time - start_time

# Choose a range of matrix sizes (n must be a power of 2).
sizes = [2**i for i in range(2, 10)]  # n = 8, 16, 32, 64, 128, 256
runtimes = []

for n in sizes:
    t = time_strassen(n)
    runtimes.append(t)
    print(f"Size: {n}x{n}, Time: {t:.6f} seconds")

# Plot the results.
plt.figure(figsize=(8, 6))
plt.plot(sizes, runtimes, 'bo-', label="Strassen Runtime")
plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.title("Runtime of Strassen’s Matrix Multiplication")
plt.grid(True)

# Plot a reference curve ~ n^(log_2 7).
n_power = np.array(sizes, dtype=float) ** (np.log2(7))
# Normalize the reference curve to start at the same point as the measured runtime.
scale_factor = runtimes[0] / (sizes[0] ** (np.log2(7)))
plt.plot(sizes, scale_factor * n_power, 'r--', label=r"$\Theta(n^{\log_2 7})$")

plt.legend()
plt.show()
plt.savefig("strassen_runtime.png")
# Estimate the exponent using a log-log plot.
log_sizes = np.log(sizes)
log_runtimes = np.log(runtimes)
slope, intercept = np.polyfit(log_sizes, log_runtimes, 1)
print(f"Estimated exponent from runtime data: {slope:.2f}")
