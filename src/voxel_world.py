import numpy as np
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage
import vnoise

def calculate_ambient_occlusion(world, size, x, y, z):
    directions = [
        (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0),
        (-1, 0, -1), (1, 0, -1), (-1, 0, 1), (1, 0, 1),
        (0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1),
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    ]
    occlusion = 0
    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
            if world[nx, ny, nz] > 0:
                occlusion += 1
    return occlusion / len(directions)

def precompute_ambient_occlusion(world, size):
    ao_matrix = np.zeros_like(world, dtype=np.float32)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if world[x, y, z] > 0:
                    ao_matrix[x, y, z] = calculate_ambient_occlusion(world, size, x, y, z)
    return ao_matrix

class NoiseGenerator:
    def __init__(self, noise_type='perlin', scale=10.0, octaves=4, persistence=0.5, lacunarity=2.0):
        self.noise_type = noise_type
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def generate_noise(self, x, y, z):
        noise = vnoise.Noise()        
        if self.noise_type == 'perlin':
            return noise.noise3(x / self.scale, y / self.scale, z / self.scale, octaves=self.octaves, persistence=self.persistence, lacunarity=self.lacunarity)
        else:
            raise ValueError("Unsupported noise type")

class VoxelWorld:
    themes = {
        'Moon': {'voxel_color': (150, 100, 150), 'light_intensity': 0.7, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Gray': {'voxel_color': (130, 130, 130), 'light_intensity': 0.8, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Rose': {'voxel_color': (180, 100, 100), 'light_intensity': 0.6, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lilac': {'voxel_color': (160, 160, 200), 'light_intensity': 0.9, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Snow': {'voxel_color': (200, 200, 250), 'light_intensity': 0.5, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Mint': {'voxel_color': (180, 255, 200), 'light_intensity': 0.7, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Peach': {'voxel_color': (255, 204, 170), 'light_intensity': 0.8, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Sky': {'voxel_color': (135, 206, 235), 'light_intensity': 0.9, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lavender': {'voxel_color': (230, 230, 250), 'light_intensity': 0.6, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lemon': {'voxel_color': (255, 255, 204), 'light_intensity': 0.5, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Ice': {'voxel_color': (200, 255, 255), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
        'Obsidian': {'voxel_color': (50, 50, 50), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
        'Mercury': {'voxel_color': (230, 230, 230), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
    }

    def __init__(self, 
                voxel_matrix, 
                theme_name='Lilac', 
                resolution=2, 
                zoom=1.0, 
                show_light_source=False, 
                time_render=False, 
                dark_bg=False, 
                color_matrix=None, 
                transparency_matrix=None, 
                specularity_matrix=None,
                transparent=False,
                singleton = None,
                singleton_color = (180, 100, 100)):

        self.size = voxel_matrix.shape[0]
        self.world = voxel_matrix

        theme = VoxelWorld.themes[theme_name]
        self.voxel_color = theme['voxel_color']
        self.light_intensity = theme['light_intensity']
        self.fog_intensity = theme['fog_intensity']
        self.light_source_position = theme['light_source_position']

        self.resolution = resolution
        self.zoom = zoom
        self.show_light_source = show_light_source
        self.time_render = time_render
        self.dark_bg = dark_bg
        self.ao_matrix = precompute_ambient_occlusion(voxel_matrix, self.size)
        self.color_matrix = color_matrix if color_matrix is not None else np.zeros((self.size, self.size, self.size, 3), dtype=np.uint8)
        self.transparency_matrix = transparency_matrix if transparency_matrix is not None else np.ones((self.size, self.size, self.size), dtype=np.float32)
        self.specularity_matrix = specularity_matrix if specularity_matrix is not None else np.zeros((self.size, self.size, self.size), dtype=np.float32)

        self.transparent = transparent if transparent is not None else False
        self.singleton = singleton if singleton is not None else None
        self.singleton_color = singleton_color if singleton_color is not None else None

    def update(self, world_attributes):
        for key, value in world_attributes.items():
            setattr(self, key, value)

    def render(self, viewing_angle=(45, 30)):
        if self.time_render:
            start_time = time.time()

        angle_x, angle_y = viewing_angle
        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)

        img_size = int(self.size * self.resolution * self.zoom * 2)
        margin = int(self.size * self.resolution * self.zoom)        
        if self.transparent or self.singleton is not None:
            bg_color = (255, 255, 255, 0) # transparent background for single voxel drawings
        else:
            bg_color = (50, 50, 50, 255) if self.dark_bg else (220, 220, 220, 255)
        image = Image.new('RGBA', (img_size + margin, img_size + margin), bg_color)
        draw = ImageDraw.Draw(image)

        voxel_center_x = self.size // 2
        voxel_center_y = self.size // 2
        voxel_center_z = self.size // 2
        center_x = (voxel_center_x - voxel_center_y) * np.cos(angle_x_rad) * self.resolution * self.zoom
        center_y = (voxel_center_x + voxel_center_y) * np.sin(angle_x_rad) * self.resolution * self.zoom - voxel_center_z * np.tan(angle_y_rad) * self.resolution * self.zoom
        offset_x = (img_size + margin) // 2 - int(center_x)
        offset_y = (img_size + margin) // 2 - int(center_y)

        if self.singleton is None:
            for x in range(self.size):
                for y in range(self.size):
                    self.render_col(draw, x, y, angle_x_rad, angle_y_rad, offset_x, offset_y)
        else:
            print('Drawing singleton')
            x, y, z = self.singleton
            self.draw_voxel(draw, x, y, z, self.singleton_color, 1, 1, angle_x_rad, angle_y_rad, offset_x, offset_y)

        if self.fog_intensity > 0 and self.singleton is None:
            image = self.apply_fog(image)

        if self.show_light_source and self.singleton is None:
            image = self.add_light_source_sphere(image, offset_x, offset_y)

        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        if self.time_render:
            elapsed_time = time.time() - start_time
            print(f"Rendering time: {elapsed_time:.2f} seconds")

        return image

    def render_col(self, draw, x, y, angle_x_rad, angle_y_rad, offset_x, offset_y):
        for z in range(self.size):
            if self.world[x, y, z] > 0:
                ao = self.ao_matrix[x, y, z]
                brightness = int((1.0 - ao) * 255 * self.light_intensity)
                base_color = tuple(min(255, int(c * brightness / 255)) for c in self.voxel_color)
                voxel_color = tuple(self.color_matrix[x, y, z]) if np.any(self.color_matrix[x, y, z]) else base_color
                transparency = self.transparency_matrix[x, y, z]
                specularity = self.specularity_matrix[x, y, z]
                self.draw_voxel(draw, x, y, z, voxel_color, transparency, specularity, angle_x_rad, angle_y_rad, offset_x, offset_y)

    def draw_voxel(self, draw, x, y, z, color, transparency, specularity, angle_x_rad, angle_y_rad, offset_x, offset_y):
        ox = int((x - y) * np.cos(angle_x_rad) * self.resolution * self.zoom + offset_x)
        oy = int((x + y) * np.sin(angle_x_rad) * self.resolution * self.zoom - z * np.tan(angle_y_rad) * self.resolution * self.zoom + offset_y)

        size = int(self.resolution * self.zoom)
        color_with_transparency = tuple(int(c * transparency) for c in color) + (int(255 * transparency),)

        draw.polygon([
            (ox, oy),
            (ox + size, oy + size // 2),
            (ox, oy + size),
            (ox - size, oy + size // 2),
        ], fill=color_with_transparency)

        draw.polygon([
            (ox, oy + size),
            (ox - size, oy + size // 2),
            (ox - size, oy + size + size // 2),
            (ox, oy + size + size // 2),
        ], fill=tuple(int(c // 2 * transparency) for c in color) + (int(255 * transparency),))

        draw.polygon([
            (ox, oy + size),
            (ox + size, oy + size // 2),
            (ox + size, oy + size + size // 2),
            (ox, oy + size + size // 2),
        ], fill=tuple(int(c // 3 * transparency) for c in color) + (int(255 * transparency),))

    def apply_fog(self, image):
        fog_overlay = Image.new('RGBA', image.size, (220, 220, 220, int(255 * self.fog_intensity * 0.5)))
        return Image.alpha_composite(image, fog_overlay)

    def add_light_source_sphere(self, image, offset_x, offset_y):
        sphere_radius = 8
        light_color = (255, 165, 0, 255)  # Bright arid orange
        light_position_x = int(self.light_source_position[0])
        light_position_y = int(self.light_source_position[1])
        draw = ImageDraw.Draw(image)
        draw.ellipse([light_position_x + offset_x - sphere_radius, light_position_y + offset_y - sphere_radius, light_position_x + offset_x + sphere_radius, light_position_y + offset_y + sphere_radius], fill=light_color)
        return image

    @staticmethod
    def show_themes():
        size = 8

        def generate_perlin_voxel_world(size):
            voxel_matrix = np.zeros((size, size, size), dtype=np.uint8)
            noise_gen = NoiseGenerator(scale=5.0)
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        if noise_gen.generate_noise(x, y, z) > 0:
                            voxel_matrix[x, y, z] = 1
            return voxel_matrix

        viewing_angle = (45, 30)
        images = []

        for theme_name in VoxelWorld.themes.keys():
            voxel_matrix = generate_perlin_voxel_world(size)
            world = VoxelWorld(voxel_matrix, theme_name, resolution=10, zoom=1.5, dark_bg=False)
            image = world.render(viewing_angle)
            images.append(image)

        # Create a 4x4 grid of subplots
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))

        # Flatten the 2D array of subplots into a 1D array
        axs = axs.flatten()

        for ax, img, theme_name in zip(axs, images, VoxelWorld.themes.keys()):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(theme_name)

        plt.show()

        plt.show()

    class Animations:
        @staticmethod
        def create_voxel_img(voxel_matrix,
                            theme_name, resolution=2,
                            viewing_angle=(45, 30),
                            zoom=1.0,
                            show_light_source=False,
                            time_render=False,
                            dark_bg=False,
                            color_matrix=None,
                            transparency_matrix=None,
                            specularity_matrix=None,
                            transparent=False,
                            singleton = None,
                            singleton_color = (180, 100, 100)):

            world = VoxelWorld(voxel_matrix, theme_name, resolution, zoom, show_light_source, time_render, dark_bg, color_matrix, transparency_matrix, specularity_matrix, transparent, singleton, singleton_color)
            image = world.render(viewing_angle)
            image = image.convert('RGBA')

            byte_stream = io.BytesIO()
            image.save(byte_stream, format='PNG')
            byte_stream.seek(0)
            return byte_stream

        @staticmethod
        def gen_gif(stack,
                    theme_name,
                    resolution=2,
                    viewing_angle=(45, 30),
                    zoom=1.0,
                    show_light_source=False,
                    dark_bg=False,
                    color_matrix_stack=None,
                    transparency_matrix_stack=None,
                    specularity_matrix_stack=None,
                    transparent=False,
                    singleton = None,
                    singleton_color = (180, 100, 100)):

            images = []
            for i, voxel_matrix in enumerate(stack):
                color_matrix = color_matrix_stack[i] if color_matrix_stack is not None else None
                transparency_matrix = transparency_matrix_stack[i] if transparency_matrix_stack is not None else None
                specularity_matrix = specularity_matrix_stack[i] if specularity_matrix_stack is not None else None
                byte_stream = VoxelWorld.Animations.create_voxel_img(voxel_matrix, theme_name, resolution, viewing_angle, zoom, show_light_source, False, dark_bg, color_matrix, transparency_matrix, specularity_matrix, transparent, singleton, singleton_color)
                image = Image.open(byte_stream)
                images.append(image)

            gif_byte_stream = io.BytesIO()
            images[0].save(gif_byte_stream, format='GIF', save_all=True, append_images=images[1:], loop=0, duration=100)
            gif_byte_stream.seek(0)
            return gif_byte_stream

        @staticmethod
        def rotate_voxel_matrix(voxel_matrix, angle):
            size = voxel_matrix.shape[0]
            rotated_matrix = np.zeros_like(voxel_matrix)
            angle_rad = np.radians(angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        if (voxel_matrix[x, y, z] > 0):
                            new_x = int(x * cos_angle - z * sin_angle)
                            new_z = int(x * sin_angle + z * cos_angle)
                            if 0 <= new_x < size and 0 <= new_z < size:
                                rotated_matrix[new_x, y, new_z] = 1
            return rotated_matrix

def display_gif(gif_byte_stream):
    display(IPImage(data=gif_byte_stream.getvalue()))
