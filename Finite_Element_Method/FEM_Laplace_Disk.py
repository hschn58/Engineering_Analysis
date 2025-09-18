import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay

# =============================================================================
# 1. Build the global stiffness matrix
# =============================================================================
def build_stiffness_matrix(nodes, triangles):
    """
    Assemble the global stiffness matrix for Laplace's equation.
    Uses linear finite elements on triangles.
    
    Parameters:
      nodes: (N,2) array of node coordinates.
      triangles: (M,3) array of indices (per triangle).
    
    Returns:
      K: Sparse global stiffness matrix in CSR format.
    """
    N = nodes.shape[0]
    I = []  # row indices for COO format
    J = []  # column indices for COO format
    V = []  # values for COO format

    # Loop over all triangles
    for tri in triangles:
        indices = tri              # local node indices for this triangle
        x = nodes[indices, 0]
        y = nodes[indices, 1]

        # Compute area of the triangle using the determinant formula:
        #  area = 0.5 * |det([[1, x1, y1],
        #                      [1, x2, y2],
        #                      [1, x3, y3]])|
        A = 0.5 * abs(np.linalg.det(np.array([[1, x[0], y[0]],
                                               [1, x[1], y[1]],
                                               [1, x[2], y[2]]])))

        # Compute coefficients for the gradients of the barycentric basis functions.
        # For a triangle with nodes (x1, y1), (x2, y2), (x3, y3), one can show:
        #   b = [y2 - y3, y3 - y1, y1 - y2]
        #   c = [x3 - x2, x1 - x3, x2 - x1]
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])

        # The local stiffness matrix entry for indices i,j is given by:
        #   Ke[i,j] = (b[i]*b[j] + c[i]*c[j])/(4*A)
        Ke = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                Ke[i,j] = (b[i]*b[j] + c[i]*c[j])/(4*A)

        # Assemble the element matrix into the global matrix
        for i_local in range(3):
            for j_local in range(3):
                I.append(indices[i_local])  # global row index
                J.append(indices[j_local])  # global column index
                V.append(Ke[i_local, j_local])  #single triangle stiffness value 
                
    # Build the global matrix (sparse, in COO format then convert to CSR)
    K = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

    print(I, J, V)
    return K

# =============================================================================
# 2. Identify boundary nodes and define the Dirichlet BC
# =============================================================================
def find_boundary_nodes(nodes, R, tol=1e-3):
    """
    Identify indices of nodes that lie on the boundary of a disk.
    
    Parameters:
      nodes: (N,2) array of node coordinates.
      R: radius of the disk.
      tol: tolerance for detecting r=R.
      
    Returns:
      boundary_nodes: array of indices for nodes on the boundary.
    """
    radii = np.sqrt(nodes[:,0]**2 + nodes[:,1]**2)
    boundary_nodes = np.where(np.abs(radii - R) < tol)[0]
    return boundary_nodes

def bc_function(x, y):
    """
    Define the boundary condition function.
    For a point (x,y) on the boundary (r=R) with theta = arctan2(y,x):
      f(R, theta) = 3/2 - 1/2*cos(2 theta)
    """
    theta = np.arctan2(y, x)
    return 1.5 - 0.5 * np.cos(2 * theta) + 1 * np.sin(5 * theta)

def apply_dirichlet_bc(K, f, nodes, bc_nodes):
    """
    Modify the system to impose Dirichlet BCs.
    
    For each boundary node i, replace the row in K so that u[i] = bc_value.
    
    Parameters:
      K: global stiffness matrix (CSR).
      f: right-hand side vector.
      nodes: array of node coordinates.
      bc_nodes: list or array of indices where BC applies.
      
    Returns:
      K_mod, f_mod: modified matrix and RHS vector.
    """
    # We create a copy since we will modify the matrix and vector.
    K = K.tolil()  # switch to LIL format for easier row modifications
    for i in bc_nodes:
        x, y = nodes[i]
        bc_val = bc_function(x, y)
        f[i] = bc_val
        # zero-out the ith row and set the diagonal to 1
        K.rows[i] = [i]
        K.data[i] = [1.0]
    K_mod = K.tocsr()
    return K_mod, f

# =============================================================================
# 3. Generate a triangular mesh on the disk
# =============================================================================
# Here we generate a set of random points inside the disk and perform Delaunay triangulation.
np.random.seed(0)
N_points = 1000
theta_rand = np.random.uniform(0, 2*np.pi, N_points)
r_rand = np.sqrt(np.random.uniform(0, 1, N_points))  # ensures uniform density in the disk
x_rand = r_rand * np.cos(theta_rand)
y_rand = r_rand * np.sin(theta_rand)
points = np.vstack((x_rand, y_rand)).T

# Create a triangulation of the points
triangulation = Delaunay(points)
nodes = points
triangles = triangulation.simplices

# =============================================================================
# 4. Assemble the finite element system
# =============================================================================
# Build the global stiffness matrix for -Δu = 0.
K = build_stiffness_matrix(nodes, triangles)

# For Laplace's equation without sources, the right-hand side is zero.
N_nodes = nodes.shape[0]
f = np.zeros(N_nodes)

# Identify the boundary nodes (where r ~ 1.0) using a tolerance.
boundary_nodes = find_boundary_nodes(nodes, R=1.0, tol=0.05)

# Apply Dirichlet boundary conditions using the given function on the boundary.
K_mod, f_mod = apply_dirichlet_bc(K, f, nodes, boundary_nodes)

# =============================================================================
# 5. Solve the linear system for the nodal values of u
# =============================================================================
u = spla.spsolve(K_mod, f_mod)

# =============================================================================
# 6. Plotting the result
# =============================================================================
# Compute the average value for each triangle
triangle_values = np.mean(u[triangles], axis=1)

plt.figure(figsize=(6,5))
plt.tripcolor(nodes[:, 0], nodes[:, 1], triangles, facecolors=triangle_values, shading='flat', cmap='rainbow', edgecolors='none')
plt.colorbar(label='u')
plt.title("Numerical solution for Laplace's equation (flat shading, coarse mesh)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.show()

