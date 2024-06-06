import matplotlib.pyplot as plt
import numpy as np

# Read data from observations.txt
data = np.loadtxt('../../data/eurepus_055.txt', delimiter=',')

# Extract columns (assuming data format)
# Modify indices according to your data format
Target = data[:, 32] * 180/np.pi
Pos = data[:, 31] * 180/np.pi
Vel = data[:,8] * 180/np.pi
Trans = data[:,4] * 180/np.pi
Trans_vel = data[:,16] * 180/np.pi

q1 = data[:,24]
q2 = data[:,25]
q3 = data[:,26]
q4 = data[:,27]


Vel_r = (data[1:,0] - data[:-1,0]) *120 * 180/np.pi

# Add more variables as needed

# Create time array (assuming data format)
time = np.arange(len(Target)) / 120  # Modify your_sampling_rate if needed

# Plot data
plt.figure(figsize=(10, 6))

# plt.plot(time, Target, label='Target')
# plt.plot(time, Pos, label='Pos')
plt.plot(time, Trans, label='FL transversal')
plt.plot(time, Trans_vel, label='FL transversal velocity')
# plt.plot(time, q1, label='q1')
# plt.plot(time, q2, label='q2')
# plt.plot(time, q3, label='q3')
# plt.plot(time, q4, label='q4')



# plt.plot(time, Vel, label='Vel filtered')
# plt.plot(time[1:], Vel_r, label='Raw vel')


# Add more plots for additional variables

plt.xlabel('Time (seconds)')
plt.ylabel('Position (degrees)')  # Modify label if needed
plt.title('Joint Positions Over Time')
plt.legend()
plt.grid(True)


# plt.show()

# Save plot as pdf
plt.savefig('../../figures/plot.pdf')
