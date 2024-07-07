import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageSequence
from IPython.display import display, Image as IPImage
import pathlib, binascii, textwrap, webbrowser
import math, json, copy, time, vnoise, random, io, os

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
        if self.noise_type == 'perlin':
            noise = vnoise.Noise()
            return np.array([[[1 if noise.noise3(x / 10.0, y / 10.0, z / 10.0) > random.uniform(-0.2, 0.2) else 0 for z in range(16)] for y in range(16)] for x in range(16)], dtype=np.uint8)
        else:
            raise ValueError("Unsupported noise type")

class Volume:
    themes = {
        'Moon': {'color': (150, 100, 150), 'light_intensity': 0.7, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Gray': {'color': (130, 130, 130), 'light_intensity': 0.8, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Rose': {'color': (180, 100, 100), 'light_intensity': 0.6, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lilac': {'color': (160, 160, 200), 'light_intensity': 0.9, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Snow': {'color': (200, 200, 250), 'light_intensity': 0.5, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Mint': {'color': (180, 255, 200), 'light_intensity': 0.7, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Peach': {'color': (255, 204, 170), 'light_intensity': 0.8, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Sky': {'color': (135, 206, 235), 'light_intensity': 0.9, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lavender': {'color': (230, 230, 250), 'light_intensity': 0.6, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Lemon': {'color': (255, 255, 204), 'light_intensity': 0.5, 'fog_intensity': 0.1, 'light_source_position': (64, 64, 128)},
        'Ice': {'color': (200, 255, 255), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
        'Obsidian': {'color': (50, 50, 50), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
        'Mercury': {'color': (230, 230, 230), 'light_intensity': 0.9, 'fog_intensity': 0.05, 'light_source_position': (64, 64, 128)},
    }

    def __init__(self, 
                voxel_matrix=None,
                viewing_angle=(30, 45),
                theme='Lilac', 
                resolution=50.0, 
                zoom=1.0, 
                show_light_source=False, 
                timeit=False, 
                dark_bg=False, 
                color_matrix=None, 
                transparency_matrix=None, 
                specularity_matrix=None,
                transparent=False):

        if voxel_matrix is None:            
            voxel_matrix = self.purlin_matrix()
        
        # 3d Numpy array
        self.world = voxel_matrix
        self.size = voxel_matrix.shape[0]

        # World lighting & color - scalars
        theme = Volume.themes[theme]
        self.color = theme['color']
        self.light_intensity = theme['light_intensity']
        self.fog_intensity = theme['fog_intensity']
        self.light_source_position = theme['light_source_position']
        self.show_light_source = show_light_source

        # Background
        self.transparent = transparent if transparent is not None else False
        self.dark_bg = dark_bg

        # World lighting & color - matrices
        self.color_matrix = color_matrix if color_matrix is not None else np.zeros((self.size, self.size, self.size, 3), dtype=np.uint8)
        self.transparency_matrix = transparency_matrix if transparency_matrix is not None else np.ones((self.size, self.size, self.size), dtype=np.float32)
        self.specularity_matrix = specularity_matrix if specularity_matrix is not None else np.zeros((self.size, self.size, self.size), dtype=np.float32)        
        self.ao_matrix = precompute_ambient_occlusion(voxel_matrix, self.size)

        # Image rendering
        self.viewing_angle = viewing_angle
        self.resolution = resolution
        self.zoom = zoom
        self.image_cache = None

        # Debug & performance
        self.timeit = timeit

    def update(self, world_attributes):
        for key, value in world_attributes.items():
            setattr(self, key, value)

    def show(self):
        self.render().show()

    def jupyter(self):
        display.Image(self.byte_stream().getvalue())

    def render(self, viewing_angle=None, timeit=False):

        if self.timeit or timeit:
            start_time = time.time()
        
        if viewing_angle is None:
            viewing_angle = self.viewing_angle

        angle_x, angle_y = viewing_angle
        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)

        img_size = int(self.size * self.resolution * self.zoom * 2)
        margin = int(self.size * self.resolution * self.zoom)        
        if self.transparent:
            bg_color = (0, 0, 0, 0) # transparent background for single voxel drawings
        else:
            # TODO: Magic color numbers
            bg_color = (50, 50, 50, 255) if self.dark_bg else (178, 189, 199, 255)
        image = Image.new('RGBA', (img_size + margin, img_size + margin), bg_color)
        draw = ImageDraw.Draw(image)

        voxel_center_x = self.size // 2
        voxel_center_y = self.size // 2
        voxel_center_z = self.size // 2
        center_x = (voxel_center_x - voxel_center_y) * np.cos(angle_x_rad) * self.resolution * self.zoom
        center_y = (voxel_center_x + voxel_center_y) * np.sin(angle_x_rad) * self.resolution * self.zoom - voxel_center_z * np.tan(angle_y_rad) * self.resolution * self.zoom
        offset_x = (img_size + margin) // 2 - int(center_x)
        offset_y = (img_size + margin) // 2 - int(center_y)

        for x in range(self.size):
            for y in range(self.size):
                self.render_col(draw, x, y, angle_x_rad, angle_y_rad, offset_x, offset_y)

        if self.fog_intensity > 0:
            image = self.apply_fog(image)

        if self.show_light_source:
            image = self.add_light_source_sphere(image, offset_x, offset_y)

        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        if self.timeit:
            elapsed_time = time.time() - start_time
            print(f"Rendering time: {elapsed_time:.2f} seconds")

        self.image_cache = image

        return image

    def render_col(self, draw, x, y, angle_x_rad, angle_y_rad, offset_x, offset_y):
        for z in range(self.size):
            if self.world[x, y, z] > 0:
                ao = self.ao_matrix[x, y, z]
                brightness = int((1.0 - ao) * 255 * self.light_intensity)
                base_color = tuple(min(255, int(c * brightness / 255)) for c in self.color)
                color = tuple(self.color_matrix[x, y, z]) if np.any(self.color_matrix[x, y, z]) else base_color
                transparency = self.transparency_matrix[x, y, z]
                specularity = self.specularity_matrix[x, y, z]
                self.draw_voxel(draw, x, y, z, color, transparency, specularity, angle_x_rad, angle_y_rad, offset_x, offset_y)

    def draw_voxel(self, draw, x, y, z, color, transparency, specularity, angle_x_rad, angle_y_rad, offset_x, offset_y):
        ox = int((x - y) * np.cos(angle_x_rad) * self.resolution * self.zoom + offset_x)
        oy = int((x + y) * np.sin(angle_x_rad) * self.resolution * self.zoom - z * np.tan(angle_y_rad) * self.resolution * self.zoom + offset_y)

        size = int(self.resolution * self.zoom)
        color_with_transparency = tuple(int(c * transparency) for c in color) + (int(255 * transparency),)
        left_face_color = tuple(int(c / 2 * transparency) for c in color) + (int(255 * transparency),)
        right_face_color = tuple(int(c / 3 * transparency) for c in color) + (int(255 * transparency),)
    
        self.voxel_size = size

        top_face = [(ox, oy), (ox + size, oy + size / 2), (ox, oy + size), (ox - size, oy + size / 2)]
        left_face = [(ox, oy + size), (ox - size, oy + size / 2), (ox - size, oy + size + size / 2), (ox, oy + size * 2)]
        right_face = [(ox, oy + size), (ox + size, oy + size / 2), (ox + size, oy + size + size / 2), (ox, oy + size * 2)]

        # Top polygon (top face of the voxel)
        draw.polygon(top_face, fill=color_with_transparency, outline='black')

        # Left polygon (left face of the voxel)
        draw.polygon(left_face, fill=left_face_color, outline='black')

        # Right polygon (right face of the voxel)
        draw.polygon(right_face, fill=right_face_color, outline='black')


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

    def add(self, foreground_world):
        """
        Overlay 2 images of the same size.
        Foreground will likely have a transparent background.
        Use to render voxel surfaces onto worlds.

        :returns: PIL Image
        """
        fg = foreground_world.render()

        if self.image_cache is not None:
            bg_cp = self.image_cache.convert('RGBA')
        else:
            bg_cp = self.render().copy() # background copy

        bg_cp.paste(fg, (0, 0), fg)

        return bg_cp

    def rotate(self, angle):
        """
        Rotate volume around its center axis
        """
        voxel_matrix = self.world
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
        self.world = rotated_matrix

    def byte_stream(self):
        """
        One-shot convenience method for visualizing (small) 3d numpy arrays.
        Returns image as a byte stream.

        Usage:
        VoxelWorld.Volume(voxel_matrix).byte_stream()
        """

        image = self.render()
        image = image.convert('RGBA')

        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_stream.seek(0)
        return byte_stream

    @staticmethod
    def purlin_matrix(size=16):
        noise = vnoise.Noise()
        return np.array([[[1 if noise.noise3(x / 10.0, y / 10.0, z / 10.0) > random.uniform(-0.2, 0.2) else 0 for z in range(size)] for y in range(size)] for x in range(size)], dtype=np.uint8)

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

        images = []

        for theme in Volume.themes.keys():
            voxel_matrix = generate_perlin_voxel_world(size)
            world = Volume(voxel_matrix, theme, resolution=10, zoom=1.5, dark_bg=False)
            image = world.render()
            images.append(image)

        # Create a 4x4 grid of subplots
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))

        # Flatten the 2D array of subplots into a 1D array
        axs = axs.flatten()

        for ax, img, theme in zip(axs, images, Volume.themes.keys()):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(theme)

        plt.show()

    def show_angles(self):

        images = []
        angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        for angle in angles:
            self.rotate(angle)
            image = self.render()
            images.append(image)

        # Create a 4x4 grid of subplots
        fig, axs = plt.subplots(3, 3, figsize=(30, 30))

        # Flatten the 2D array of subplots into a 1D array
        axs = axs.flatten()

        for ax, img, angle in zip(axs, images, angles):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(str(angle))

        plt.show()

    @staticmethod
    def merge_matrices(matrix1, matrix2):
        # Ensure both matrices are of the same size
        assert matrix1.shape == matrix2.shape, "Matrices must be of the same size"

        # Create a result matrix by prioritizing non-zero elements from the first matrix
        result_matrix = np.where(matrix1 != 0, matrix1, matrix2)

        return result_matrix    

class Axes(Volume):
    """
    World axis helpers
    """
    def __init__(self, volume, corners=True):
        self.__dict__.update(volume.__dict__)
        size_x, size_y, size_z = self.world.shape

        matrix = np.zeros((size_x, size_y, size_z), dtype=int)

        matrix[:, 0, 0] = 1  # x-axis
        matrix[0, :, 0] = 1  # y-axis
        matrix[0, 0, :] = 1  # z-axis

        # Create a 3D matrix for colors with an additional dimension for RGB values
        # Using dtype=object to store tuples
        color_matrix = np.zeros((size_x, size_y, size_z), dtype=object)

        # Initialize all elements to 0
        color_matrix[:, :, :] = 0

        # Set RGB values along the x, y, and z axes
        for i in range(min(size_x, size_y, size_z)):
            color_matrix[i, 0, 0] = (254, 1, 1)  # x-axis
            color_matrix[0, i, 0] = (1, 254, 1)  # y-axis
            color_matrix[0, 0, i] = (1, 1, 254)  # z-axis

        self.transparent = True
        self.world = matrix
        self.color_matrix = color_matrix

        if corners:
            corners = Corners(self)
            self.world = self.merge_matrices(corners.world, self.world)
            self.color_matrix = self.merge_matrices(corners.color_matrix, self.color_matrix)

class Corners(Volume):
    """
    Add corner markers (colorblind-friendly)
    """
    def __init__(self, volume):
        self.__dict__.update(volume.__dict__)
        size_x, size_y, size_z = self.world.shape

        matrix = np.zeros((size_x, size_y, size_z), dtype=int)

        # Set 1s in each corner
        matrix[0, 0, 0] = 1
        matrix[0, 0, -1] = 1
        matrix[0, -1, 0] = 1
        matrix[0, -1, -1] = 1
        matrix[-1, 0, 0] = 1
        matrix[-1, 0, -1] = 1
        matrix[-1, -1, 0] = 1
        matrix[-1, -1, -1] = 1

        color_matrix = np.zeros((size_x, size_y, size_z), dtype=object)
    
        color_matrix[0, 0, 0] = (220, 162, 55)
        color_matrix[0, 0, -1] = (111, 178, 228)
        color_matrix[0, -1, 0] = (70, 156, 118)
        color_matrix[0, -1, -1] = (238, 228, 97)
        color_matrix[-1, 0, 0] = (48, 112, 173)
        color_matrix[-1, 0, -1] = (193,	125, 165)
        color_matrix[-1, -1, 0] = (193,	125, 165)
        color_matrix[-1, -1, -1] = (0, 0, 0)

        self.transparent = True
        self.world = matrix
        self.color_matrix = color_matrix

class Vixel(Volume):
    """
    Render numpy-defined voxel worlds with Vixel
    """    
    def __init__(self, volume):
        self.__dict__.update(volume.__dict__)

    @staticmethod
    def spherical_to_cartesian(azimuthal, elevation, r=1):
        azimuthal = math.radians(azimuthal)
        elevation = math.radians(elevation)
        x = r * math.sin(azimuthal) * math.cos(elevation)
        y = r * math.sin(azimuthal) * math.sin(elevation)
        z = r * math.cos(azimuthal)
        return x, y, z

    def __generate_javascript(self):

        # Vixel 
        world = np.swapaxes(self.world, 2, 1)
        color_matrix = np.swapaxes(self.color_matrix, 2, 1)

        # Serialize ones matrix to a JSON string
        ones_matrix_str = json.dumps(world.tolist())

        # Serialize color matrix
        color_matrix_str = json.dumps(color_matrix.tolist())

        js_string = textwrap.dedent('''

            /* 🟢 EXTRACT MATRIX */
            const jsonOnesMatrix = '{0}';
            const matrix = JSON.parse(jsonOnesMatrix);
            console.log('3D array:', matrix);
        
            const jsonColorMatrix = '{1}'
            const colorMatrix = JSON.parse(jsonColorMatrix);
            console.log('color array:', colorMatrix);

            /* 🟢 INIT VIXEL */
            const bounds = [{2}, {3}, {4}];
            const vixel = new Vixel(canvas, ...bounds);

            /* 🟢 CAMERA */
            vixel.camera(
                [60, 50, 50], // Camera position
                [0.25, -0.785, 0.5], // Camera target
                [0, 1, 0], // Up
                Math.PI / 4 // Field of view
            );

            /* 🟢 DEPTH OF FIELD AND SUNLIGHT */
            vixel.dof(0.5, 0.25);
            vixel.sun(-17, (1.5 * Math.PI) / 2, -1, 0.5);

            /* 🟢 SET VIXELS */
            // Bounds of the matrix
            const [matrixX, matrixY, matrixZ] = [matrix.length, matrix[0].length, matrix[0][0].length];

            const baseColor = [12, 12, 24];

            // Set voxels based on the matrix values
            for (let x = 0; x < matrixX; x++) {{
                for (let y = 0; y < matrixY; y++) {{
                    for (let z = 0; z < matrixZ; z++) {{
                        if (matrix[x][y][z] === 1) {{
                            const color = getColor(colorMatrix, x, y, z, baseColor);
                            vixel.set(x, y, z, {{
                                red: color[0],
                                green: color[1],
                                blue: color[2],
                            }});
                        }}
                    }}
                }}
            }}

            /* 🟢 RENDERING LOOP (RECURSIVE) */
            let samples = 0;
            function renderLoop() {{
                vixel.sample(2);
                vixel.display();
                samples += 2;
                if (samples < 2048) {{
                    requestAnimationFrame(renderLoop);
                }}
            }}
            renderLoop();
        '''.format(
            ones_matrix_str,
            color_matrix_str,
            self.world.shape[0],
            self.world.shape[1],
            self.world.shape[2]
        ))

        return js_string

    def html(self, filename='vixel.html', auto_open=True, return_html=False):
        """
        Save a standalone HTML file with a Vixel rendering.
        """

        # Should return path of this module (eg /VoxelWorld/src)
        p = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

        # Navigate to HTML template file
        template_path = pathlib.Path(*p.parts[:-1] + ('vixel','docs','index.html'))

        with template_path.open(mode="r", encoding="utf-8") as html_file:
            html = html_file.read()

        html = html.replace(
            '/*** 龴ↀ◡ↀ龴 REPLACE THIS LINE 龴ↀ◡ↀ龴 ***/', 
            self.__generate_javascript()
        )

        # TODO: Revisit for Windows and internationalization
        downloads_path = pathlib.Path(*pathlib.Path.home().parts + ('Downloads', filename))

        downloads_path.write_text(html)

        if auto_open:
            webbrowser.open('file://' + str(downloads_path), new = 0)

        if return_html:
            return html


class Surface(Volume):
    def __init__(self, volume):
        """
        Initialize a Surface from a Volume instance.
        """
        self.__dict__.update(volume.__dict__)
        self.size_x, self.size_y, self.size_z = self.world.shape

        self.transparent = True # In general, surface renders won't have backgrounds
                                # because they will be overlaid on volume renders

        voxel_world_surface = np.zeros((self.size_x, self.size_y), dtype=int)

        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z - 1, -1, -1):
                    if self.world[x, y, z] > 0:
                        voxel_world_surface[x, y] = z + 1
                        break

        self.topology = voxel_world_surface

        self.world = self.surface_world()
    
    def surface_world(self, mask3d=None):
        """
        Returns a 3d numpy surface defined by a 2d (x, y) matrix of z values.
        The surface thickness does not exceed a single voxel.
        """
        surface_world = np.zeros(self.world.shape)

        matrix = self.topology

        # Build the surface in 3d space
        for i in range(matrix.shape[0]):  # Iterate over rows
            for j in range(matrix.shape[1]):  # Iterate over columns
                surface_world[i, j, int(matrix[i, j])-1] = 1
    
        # Apply the 3d mask
        if mask3d is not None:
            nz_X, nz_Y, nz_Z = np.nonzero(mask3d)         
            surface_world = surface_world * mask3d

        return surface_world

class Agent(Surface):
    def __init__(self, surface, mask=None, color=(255, 0, 0)):
        """
        Initialize a surface-inhabiting agent from a Surface instance.
        Usage: agent = Agent(surf)
        """
        self.__dict__.update(surface.__dict__)

        if mask is None:
            # Agent defaults to entire surface
            mask = np.ones(self.topology.shape)
            mask3d = np.ones(self.world.shape)
        else:
            assert(mask.shape == self.topology.shape)
            mask3d = np.zeros(self.world.shape)

            allowed_z_values = mask * self.topology
            for i in range(allowed_z_values.shape[0]):
                for j in range(allowed_z_values.shape[1]):
                    if allowed_z_values[i, j] > 0:
                        mask3d[i, j, allowed_z_values[i, j]-1] = 1

        self.color = color
        self.mask = mask
        self.world = self.surface_world(mask3d)

    def cell(self, x, y, color=(255, 0, 0)):
        """
        Create a world for a single voxel cell
        Usage: Place a single voxel on the world surface at coordinates 10,10:
        agent = Agent(surf).cell(10,10)
        """
        mask3d = np.zeros(self.world.shape)
        mask3d[x, y, self.topology[x, y]-1] = 1

        mask = np.zeros(self.topology.shape)
        mask[x, y] = 1

        self.color = color
        self.mask = mask
        self.world = self.surface_world(mask3d)

        return self

class Sequence:
    def __init__(self, worlds, timeit=False):
        """
        Usage:

        from voxel_world import Sequence, Volume
        
        volumes = [Volume(viewing_angle=(45, 30+(i*10))) for i in range(10)]    

        Sequence(volumes).render()
        """

        self.worlds = worlds
        self.timeit = timeit
        self.frames = [world.render() for world in self.worlds]
    
    def apply_bg(self, bg):
        """
        Apply a background world to a Sequence
        """

        # Understand if background is an image, Sequence of images, etc
        match type(bg).__name__:
            case 'Sequence':
                # TODO For dynamic backgrounds
                pass            
            case 'Volume':
                # For static backgrounds
                bg_im = bg.render() # background copy

                new_frames = []
                for i, frame in enumerate(self.frames):
                    frame_w_background = bg_im.copy()
                    frame_w_background.paste(frame, (0, 0), frame)
                    new_frames.append(frame_w_background)

        seq_w_bg = copy.deepcopy(self)
        seq_w_bg.frames = new_frames

        return seq_w_bg
       
    def render(self):
        """
        Return a GIF byte stream from self.frames
        """
        gif_stream = self.frames_to_gif_stream()
        return gif_stream

    def frames_to_gif_stream(self):
        frames = self.frames
        gif_stream = io.BytesIO()
        frames[0].save(gif_stream, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=100)
        gif_stream.seek(0)
        return gif_stream

    def save(self, filename='voxel_animation.gif'):
        frames = self.frames
        frames[0].save(filename, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=100)

    def jupyter(self):
        gif_stream = self.frames_to_gif_stream()
        display(IPImage(data=gif_stream.getvalue()))
    
    def mpl(self):
        fig, ax = plt.subplots()
        ims = []
        for frame in self.frames:
            ims.append([plt.imshow(np.asarray(frame))])
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
        plt.show()

    @staticmethod
    def random_walk(num_steps=500, grid_size=16):
        """
        Returns coordinates for random walk Agent demo

        Usage:
        from voxel_world import Volume, Surface, Agent
        surf = Surface(Volume())
        agents = [Agent(surf).cell(x, y) for x, y in Sequence.random_walk()]
        """
        # Initialize the starting point (e.g., the center of the grid)
        start_point = (grid_size // 2, grid_size // 2)

        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Initialize the list of walk coordinates with the starting point
        walk_coordinates = [start_point]

        # Generate the random walk
        current_point = start_point
        for _ in range(num_steps):
            move = random.choice(moves)
            next_point = (current_point[0] + move[0], current_point[1] + move[1])
            
            # Ensure the next point is within the grid boundaries
            if 0 <= next_point[0] < grid_size and 0 <= next_point[1] < grid_size:
                walk_coordinates.append(next_point)
                current_point = next_point

        return walk_coordinates
    
    @staticmethod
    def snake(num_steps=500, grid_size=16):
        """
        Returns coordinates for snake Agent demo

        Usage:
        from voxel_world import Volume, Surface, Agent
        surf = Surface(Volume())
        agents = [Agent(surf, mask) for mask in Sequence.snake()]
        """
        
        # Initialize the grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Initialize the snake as a list of tuples (starting in the center)
        snake = [(grid_size // 2, grid_size // 2)]
        grid[snake[0]] = 1

        # Function to place food on the grid
        def place_food(grid, snake):
            while True:
                food_position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
                if food_position not in snake:
                    grid[food_position] = 1
                    return food_position

        # Place the first food
        food_position = place_food(grid, snake)

        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Initialize a list to store the evolution of the grid
        grid_evolution = [grid.copy()]

        # Function to get the next move towards the food
        def get_next_move(snake, food_position):
            head = snake[0]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            valid_moves = []
            
            for direction in directions:
                next_position = (head[0] + direction[0], head[1] + direction[1])
                if (0 <= next_position[0] < grid_size and 
                    0 <= next_position[1] < grid_size and 
                    next_position not in snake):
                    valid_moves.append((next_position, direction))
            
            if not valid_moves:
                return None, None
            
            # Choose the move that minimizes the Manhattan distance to the food
            next_move = min(valid_moves, key=lambda x: abs(x[0][0] - food_position[0]) + abs(x[0][1] - food_position[1]))
            return next_move

        # Simulate the game of Snake
        for _ in range(num_steps):
            next_position, move = get_next_move(snake, food_position)
            if next_position is None:
                # No valid moves, game over
                break

            # Move the snake
            snake.insert(0, next_position)
            if next_position == food_position:
                # Place new food if the snake eats the current food
                food_position = place_food(grid, snake)
            else:
                # Remove the tail if no food is eaten
                tail = snake.pop()
                grid[tail] = 0
            
            # Update the grid with the new snake position
            grid[next_position] = 1
            
            # Store the current state of the grid
            grid_evolution.append(grid.copy())
        
        return grid_evolution
