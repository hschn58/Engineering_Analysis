import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Parameters
L = np.pi  # Period of the function
delta_initial = L/100  # Initial delta value

# Function to compute coefficients for the left cubic polynomial
def compute_left_coefficients(L, delta):
    h = delta
    # Setup equations based on boundary conditions
    # At s = 0: f = 0, f' = 0
    # At s = h: f = -L + h, f' = 1
    # Let s = x + L
    # Polynomial: f(s) = a*s^3 + b*s^2 + c*s + d
    # Since c = 0, d = 0, we have:
    # Equation 1: a*h^3 + b*h^2 = -L + h
    # Equation 2: 3*a*h^2 + 2*b*h = 1

    # Create the coefficient matrix and RHS vector
    A = np.array([
        [h**3, h**2],
        [3*h**2, 2*h]
    ])
    B = np.array([-L + h, 1])

    # Solve for a and b
    a_left, b_left = np.linalg.solve(A, B)

    c_left = 0
    d_left = 0
    return a_left, b_left, c_left, d_left

# Function to compute coefficients for the right cubic polynomial
def compute_right_coefficients(L, delta):
    h = delta
    # At s = 0: f = L - h, f' = 1
    # At s = h: f = 0, f' = 0
    # Let s = x - (L - h)
    # Polynomial: f(s) = a*s^3 + b*s^2 + c*s + d
    # Since d = L - h, c = 1, we have:
    # Equation 1: a*h^3 + b*h^2 + c*h + d = 0
    # Equation 2: 3*a*h^2 + 2*b*h + c = 0

    # Known values
    c_right = 1
    d_right = L - h

    # Create the coefficient matrix and RHS vector
    A = np.array([
        [h**3, h**2],
        [3*h**2, 2*h]
    ])
    B = np.array([-c_right*h - d_right, -c_right])

    # Solve for a and b
    a_right, b_right = np.linalg.solve(A, B)
    return a_right, b_right, c_right, d_right

# Define the function f(x)
def f(x, delta):
    fx = np.zeros_like(x)
    # Left segment
    mask_left = (x >= -L) & (x <= -L + delta)
    if np.any(mask_left):
        a_left, b_left, c_left, d_left = compute_left_coefficients(L, delta)
        s_left = x[mask_left] + L  # s ranges from 0 to delta
        fx[mask_left] = a_left * s_left**3 + b_left * s_left**2 + c_left * s_left + d_left

    # Middle segment
    mask_middle = (x > -L + delta) & (x < L - delta)
    fx[mask_middle] = x[mask_middle]

    # Right segment
    mask_right = (x >= L - delta) & (x <= L)
    if np.any(mask_right):
        a_right, b_right, c_right, d_right = compute_right_coefficients(L, delta)
        s_right = x[mask_right] - (L - delta)  # s ranges from 0 to delta
        fx[mask_right] = a_right * s_right**3 + b_right * s_right**2 + c_right * s_right + d_right

    return fx

# Number of Fourier coefficients
N_max = 200  # Adjust as needed

# Discretize x for numerical integration
N_integration = 2000
x_integration = np.linspace(-L, L, N_integration)

# Compute Fourier coefficients numerically
def compute_coefficients(N_terms, delta):
    fx_integration = f(x_integration, delta)
    a0 = (1 / L) * np.trapz(fx_integration, x_integration)
    an = np.zeros(N_terms)
    bn = np.zeros(N_terms)
    for n in range(1, N_terms + 1):
        cos_nx = np.cos(n * np.pi * x_integration / L)
        sin_nx = np.sin(n * np.pi * x_integration / L)
        an[n - 1] = (1 / L) * np.trapz(fx_integration * cos_nx, x_integration)
        bn[n - 1] = (1 / L) * np.trapz(fx_integration * sin_nx, x_integration)
    return a0, an, bn

# Partial sum of the Fourier series
def fourier_series(a0, an, bn, x, N):
    result = np.zeros_like(x) + a0 / 2
    for n in range(1, N + 1):
        result += (an[n - 1] * np.cos(n * np.pi * x / L) +
                   bn[n - 1] * np.sin(n * np.pi * x / L))
    return result

# Prepare the figure
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
x_plot = np.linspace(-L, L, 1000)
delta_slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03])
delta_slider = Slider(delta_slider_ax, 'Delta', 0.01, L / 2, valinit=delta_initial)

# Initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Animation function
def animate(N):
    N_terms = N + 1
    delta = delta_slider.val
    fx = f(x_plot, delta)
    a0, an, bn = compute_coefficients(N_terms, delta)
    S_N = fourier_series(a0, an, bn, x_plot, N_terms)
    line1.set_data(x_plot, fx)
    line2.set_data(x_plot, S_N)
    ax.set_title(f'Fourier Approximation with N = {N_terms} terms')
    return line1, line2

# Update function for the slider
def update(val):
    ani.event_source.stop()
    ani.event_source.start()

delta_slider.on_changed(update)

# Initial plot
delta = delta_slider.val
fx = f(x_plot, delta)
line1, = ax.plot(x_plot, fx, 'k', label='Modified Sawtooth Function')
line2, = ax.plot([], [], 'r', label='Fourier Approximation')
ax.set_xlim(-L, L)
ax.set_ylim(-1.2 * L, 1.2 * L)
ax.legend()

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=N_max, init_func=init,
                              interval=200, blit=True)

plt.show()
