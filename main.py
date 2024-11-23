import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Constants for the double pendulum
m1 = 1.0  # Mass of pendulum 1 (kg)
m2 = 1.0  # Mass of pendulum 2 (kg)
r1 = 1.0  # Length of pendulum 1 (m)
r2 = 1.0  # Length of pendulum 2 (m)
g = 9.81  # Gravitational acceleration (m/s^2)

# Initial conditions: [theta1, omega1, theta2, omega2]
theta1_0 = 0.0  # Initial angle of pendulum 1 (rad)
omega1_0 = 0.0       # Initial angular velocity of pendulum 1 (rad/s)
theta2_0 = np.pi  # Initial angle of pendulum 2 (rad)
omega2_0 = 0.0       # Initial angular velocity of pendulum 2 (rad/s)

initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]

# Time span for the simulation
t_end = 100                            # End time (s)
t_span = (0, t_end)                   # Time span for the simulation
start_time = time.time()

# Function to compute the state-space derivatives
def double_pendulum_state_space(t, state):
    theta1, omega1, theta2, omega2 = state

    # Mass matrix
    M = np.array([
        [1, (m2*r2)/((m1+m2)*r1)],
        [(r1/r2)*np.cos(theta1-theta2), 1]
    ])

    # Right-hand side of the equations
    F = np.array([
        -(g/r1)*np.sin(theta1) - (m2*r2/((m1+m2)*r1))*omega2*np.sin(theta1-theta2),
        (r1/r2)*omega1**2*np.sin(theta1-theta2) - (g/r2)*np.sin(theta2)
    ])

    # Solve for angular accelerations
    accelerations = np.linalg.solve(M, F)
    omega1_dot, omega2_dot = accelerations

    # Return the state derivatives
    return [omega1, omega1_dot, omega2, omega2_dot]

# Solve the system using solve_ivp
solution = solve_ivp(
    double_pendulum_state_space,
    t_span,
    initial_state,
    rtol=1e-12,
    atol=1e-12,
    method='RK45'
)

# Extract the results
theta1 = solution.y[0]
theta2 = solution.y[2]

# Define positions of the pendulum bobs
def get_positions(theta1, theta2):
    x1 = r1 * np.sin(theta1)
    y1 = -r1 * np.cos(theta1)
    x2 = x1 + r2 * np.sin(theta2)
    y2 = y1 - r2 * np.cos(theta2)
    return x1, y1, x2, y2

# Create figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2 * (r1 + r2), 2 * (r1 + r2))
ax.set_ylim(-2 * (r1 + r2), 2 * (r1 + r2))
ax.set_aspect('equal', adjustable='box')
ax.set_title("Double Pendulum Animation")

# Initialize lines for rods and points for masses
line, = ax.plot([], [], 'o-', lw=2, markersize=8)
trace, = ax.plot([], [], 'r-', lw=0.5, alpha=0.7)  # Trace path of the second pendulum

# Storage for trace of second pendulum
x2_trace, y2_trace = [], []

# Initialization function for animation
def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

# Synchronize the animation to `solution.t` for accurate real-time playback
def update_real_time(frame):
    global x2_trace, y2_trace

    # Ensure frame is within valid range
    if frame >= len(solution.t):
        return line, trace

    # Current angles
    theta1_frame = theta1[frame]
    theta2_frame = theta2[frame]

    # Get positions of pendulum bobs
    x1, y1, x2, y2 = get_positions(theta1_frame, theta2_frame)

    # Update pendulum rods and masses
    line.set_data([0, x1, x2], [0, y1, y2])

    # Update trace of the second pendulum bob
    x2_trace.append(x2)
    y2_trace.append(y2)
    trace.set_data(x2_trace, y2_trace)

    return line, trace

# Use `solution.t` as the interval for accurate timing
real_time_intervals = np.diff(solution.t) * 1000  # Convert time differences to milliseconds
real_time_intervals = np.append(real_time_intervals, real_time_intervals[-1])  # Repeat last interval for stability

# Start time to synchronize with `solution.t`
start_time = time.time()

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update_real_time,
    frames=len(solution.t),
    init_func=init,
    blit=True,
    interval=10,  # Approximate interval; real-time sync managed dynamically
    repeat=False
)

plt.show()


x1, y1, x2, y2 = get_positions(theta1, theta2)

#plot the results in x-y plane
plt.plot(x1, y1, label='pendulum1')
plt.plot(x2, y2, label='pendulum2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

theta1_wrapped = (theta1 + np.pi) % (2 * np.pi) - np.pi
theta2_wrapped = (theta2 + np.pi) % (2 * np.pi) - np.pi

print(solution.t)

# plot the results in theta vs time
plt.plot(solution.t, theta1_wrapped, label='theta1')
plt.plot(solution.t, theta2_wrapped, label='theta2')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.legend()

plt.yticks(
    ticks=np.arange(-np.pi, np.pi + np.pi / 2, np.pi / 2),  # Tick marks from -π to π
    labels=[r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
)
plt.show()

# plot theta1 vs theta2
plt.plot(theta1_wrapped, theta2_wrapped)
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.show()

