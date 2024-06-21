import numpy as np; import io; import vnoise
from IPython.display import display, Image as IPImage
from voxel_world import Volume

# Define the size of the voxel world
size = 32

# Create a voxel matrix with Perlin noise
voxel_matrix = np.zeros((size, size, size), dtype=np.uint8)
scale = 10.0
threshold = 0.1

noise = vnoise.Noise()

for x in range(size):
    for y in range(size):
        for z in range(size):
            if noise.noise3(x / scale, y / scale, z / scale, octaves=4) > threshold:
                voxel_matrix[x, y, z] = 1

# Create a color matrix with Perlin noise
color_matrix = np.zeros((size, size, size, 3), dtype=np.uint8)
for x in range(size):
    for y in range(size):
        for z in range(size):
            noise_value = noise.noise3(x / scale, y / scale, z / scale, octaves=4)
            color = int((noise_value + 1) * 127)  # Normalize noise value to [0, 255]
            color_matrix[x, y, z] = (color, 255 - color, (color * 2) % 256)

# Initialize VoxelWorld with the voxel and color matrices
world = Volume(voxel_matrix, theme='Lilac', resolution=10, zoom=2.0, show_light_source=False, dark_bg=False, color_matrix=color_matrix)

# Render and display the voxel world
image = world.render()
byte_stream = io.BytesIO()
image.save(byte_stream, format='PNG')
byte_stream.seek(0)
display(IPImage(data=byte_stream.getvalue()))