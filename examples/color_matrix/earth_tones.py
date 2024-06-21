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

# Create a color matrix with earth-tone, natural colors
color_matrix = np.zeros((size, size, size, 3), dtype=np.uint8)
for x in range(size):
    for y in range(size):
        for z in range(size):
            noise_value = noise.noise3(x / scale, y / scale, z / scale, octaves=4)
            r = int((noise_value + 1) * 100 + 50)  # Brownish red
            g = int((np.sin(x / 5.0) + 1) * 60 + 60)  # Earthy green
            b = int((np.cos(y / 5.0) + 1) * 40 + 40)  # Natural blue/grey
            color_matrix[x, y, z] = (r, g, b)

# Initialize VoxelWorld with the voxel and color matrices
world = Volume(voxel_matrix, theme='Lilac', resolution=10, zoom=2.0, show_light_source=False, dark_bg=True, color_matrix=color_matrix)

# Function to rotate the voxel matrix around its center
def rotate_voxel_matrix(voxel_matrix, angle):
    size = voxel_matrix.shape[0]
    rotated_matrix = np.zeros_like(voxel_matrix)
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    center = size // 2

    for x in range(size):
        for y in range(size):
            for z in range(size):
                if voxel_matrix[x, y, z] > 0:
                    new_x = int((x - center) * cos_angle - (y - center) * sin_angle + center)
                    new_y = int((x - center) * sin_angle + (y - center) * cos_angle + center)
                    if 0 <= new_x < size and 0 <= new_y < size:
                        rotated_matrix[new_x, new_y, z] = voxel_matrix[x, y, z]
    return rotated_matrix

# Function to rotate the color matrix around its center
def rotate_color_matrix(color_matrix, angle):
    size = color_matrix.shape[0]
    rotated_matrix = np.zeros_like(color_matrix)
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    center = size // 2

    for x in range(size):
        for y in range(size):
            for z in range(size):
                new_x = int((x - center) * cos_angle - (y - center) * sin_angle + center)
                new_y = int((x - center) * sin_angle + (y - center) * cos_angle + center)
                if 0 <= new_x < size and 0 <= new_y < size:
                    rotated_matrix[new_x, new_y, z] = color_matrix[x, y, z]
    return rotated_matrix

# Function to create rotating frames
def create_rotating_frames(world, steps=36):
    frames = []
    for i in range(steps):
        angle = (360 / steps) * i
        rotated_voxel_matrix = rotate_voxel_matrix(voxel_matrix, angle)
        rotated_color_matrix = rotate_color_matrix(color_matrix, angle)
        image = Volume(rotated_voxel_matrix, 'Lilac', resolution=10, zoom=2.0, dark_bg=True, color_matrix=rotated_color_matrix).render(viewing_angle=(45,30))
        frames.append(image)
    return frames

# Generate frames for rotation
frames = create_rotating_frames(world, steps=36)

# Create a GIF animation with slower speed
gif_byte_stream = io.BytesIO()
frames[0].save(gif_byte_stream, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=200)
gif_byte_stream.seek(0)

# Display the GIF animation
display(IPImage(data=gif_byte_stream.getvalue()))