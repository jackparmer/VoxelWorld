import numpy as np
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage
from noise import pnoise3
from voxel_world import VoxelWorld

def create_voxel_sphere(size, radius):
    """Creates a spherical voxel object within a cubic voxel world."""
    voxel_matrix = np.zeros((size, size, size), dtype=np.uint8)
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 < radius**2:
                    voxel_matrix[x, y, z] = 1
    return voxel_matrix

def create_specularity_matrix(size, radius):
    """Creates a specularity matrix with higher values near the center of the sphere."""
    specularity_matrix = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                distance_to_center = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                if distance_to_center < radius:
                    specularity_matrix[x, y, z] = 1.0 - distance_to_center / radius
    return specularity_matrix

def animate_light_source_orbit(voxel_matrix, specularity_matrix, size=32, steps=50):
    """Creates an animation showing the light source making a full orbit around the voxel planet."""
    images = []
    center = size // 2
    orbit_radius = int(size * 5)

    for step in range(steps):
        angle = (2 * np.pi / steps) * step
        light_position = (
            center + int(orbit_radius * np.cos(angle)),
            center + int(orbit_radius * np.sin(angle)),
            center - size/2
        )
        light_intensity = 0.5 + 0.5 * np.sin(angle)  # Vary intensity with position

        world = VoxelWorld(voxel_matrix, 'Lilac', resolution=10, zoom=1.5, show_light_source=True, dark_bg=True, specularity_matrix=specularity_matrix)
        world.light_source_position = light_position
        world.light_intensity = light_intensity
        image = world.render(viewing_angle=(45, 30))
        images.append(image)

    gif_byte_stream = io.BytesIO()
    images[0].save(gif_byte_stream, format='GIF', save_all=True, append_images=images[1:], loop=0, duration=100)
    gif_byte_stream.seek(0)
    return gif_byte_stream

def display_gif(gif_byte_stream):
    display(IPImage(data=gif_byte_stream.getvalue()))

# Example usage:
size = 32
radius = 12

# Create a voxel sphere
voxel_matrix = create_voxel_sphere(size, radius)

# Create a specularity matrix for the sphere
specularity_matrix = create_specularity_matrix(size, radius)

# Animate the light source orbiting around the voxel planet
gif_byte_stream = animate_light_source_orbit(voxel_matrix, specularity_matrix, size=size, steps=50)

# Display the animation
display_gif(gif_byte_stream)
