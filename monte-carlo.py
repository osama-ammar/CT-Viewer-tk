import numpy as np
import matplotlib.pyplot as plt

def simulate_energy_deposition(ct_image, beam_position, beam_energy, num_particles=1000):
    # Constants
    pixel_size = 1  # Assume each pixel represents 1 cm^2 (simplified)
    
    # Initialize dose map
    dose_map = np.zeros_like(ct_image, dtype=float)

    for _ in range(num_particles):
        # Initial position of the particle (assumed to start from the left side)
        particle_position = [150, np.random.randint(0, ct_image.shape[1])]

        while 0 <= particle_position[0] < ct_image.shape[0] and 0 <= particle_position[1] < ct_image.shape[1]:
            # Calculate dose based on pixel attenuation
            pixel_value = ct_image[tuple(particle_position)]
            attenuation_factor = np.exp(-pixel_value * pixel_size)  # Simplified attenuation model

            # Update dose map
            dose_map[tuple(particle_position)] += beam_energy * attenuation_factor

            # Move the particle to the next position (move to the right)
            particle_position[0] += 1

    return dose_map

# Generate a simple 128x128 CT image (for illustration purposes)
ct_image = np.load("volume.npy")[100]




# Simulate energy deposition
beam_position = 10  # Assume the beam is applied at the 10th column
beam_energy = 0.005 # Example beam energy in MeV
simulated_dose_map = simulate_energy_deposition(ct_image, beam_position, beam_energy)

# Display the CT image and simulated dose map
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(ct_image, cmap='gray', origin='lower')
plt.title('CT Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(simulated_dose_map, cmap='hot', origin='lower')
plt.title('Simulated Dose Map')
plt.colorbar()

plt.show()