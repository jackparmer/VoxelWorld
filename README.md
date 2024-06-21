# VoxelWorld
Create delicious Voxel worlds in Python

[Live demo on Py.Cafe ☕](https://py.cafe/jackparmer/voxel-worlds)

<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/680dcde4-299e-4508-8cb7-1779831b1b98">

<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/917f38ac-dd26-4419-9725-0693ca05aaa1">

## Install

### From PyPi

> pip3 install voxel_world

### From source

```sh
git clone https://github.com/jackparmer/VoxelWorld.git
cd VoxelWorld
python3 -m pip install .

from voxel_world import VoxelWorld
```

## About

For physics simulation, games, art, and fun

Inspo: https://github.com/wwwtyro/vixel

Features!
- Automatic GIF generation
- Numpy 3d ones array in -> Voxel world out
- Fast-ish (as fast as rendering on the CPU can be)
- Portable! Outputs simple image files
- Notebooks! Works well in the Jupyter notebook ecosystem
- Eye candy! [Ambient occlusion](https://en.wikipedia.org/wiki/Ambient_occlusion), specularity, etc

Known issues (TODO)
- Speed: Need to migrate to a GPU-based renderer while maintaining portability (suggestions?)
- Illumination: Light source ray tracing is wonky - but you can fake it (see light_source.py example)
- Cut offs: The bottom of some voxel cubes are cut off - I'm not sure why
- Likely much more...

***

# Examples

## Animations

```py
from voxel_world import Volume, Surface, Agent, Sequence

volume = Volume(Volume.purlin_matrix(64));
surf = Surface(volume);
agents = [Agent(surf, mask) for mask in Sequence.snake(surf.topology, grid_size=64, num_steps=1000)];
seq = Sequence(agents);

seq2 = seq.apply_bg(volume)

seq2.save('voxel_animation64_v2.gif')
```
<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/blob/main/voxel_animation64.gif?raw=true">

## Surfaces API

```
from voxel_world import Volume, Surface;
volume = Volume(Volume.purlin_matrix(32)); surf = Surface(volume)
surf.color = (255,0,0)
volume.add(surf).show()
```
![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/30d0d2f8-8f7b-426c-b394-d18ca2c47c93)

## Randomly generated worlds

[Live demo on Py.Cafe ☕](https://py.cafe/jackparmer/voxel-worlds)

```py
import random; import vnoise
from IPython.display import display, Image as IPImage
from voxel_world import Volume

noise = vnoise.Noise()

# Display in Juypter
display(IPImage(Volume(
    np.array([[[1 if noise.noise3(x / 10.0, y / 10.0, z / 10.0) > random.uniform(-0.2, 0.2) else 0 for z in range(16)] for y in range(16)] for x in range(16)], dtype=np.uint8),
    theme=random.choice(list(Volume.themes.keys())),
    resolution=10,
    viewing_angle=(random.randint(0, 90), random.randint(0, 90)),
    zoom=2.0,
    show_light_source=False,
    dark_bg=False
).byte_stream().getvalue()))
```

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/25bd612e-b8e9-42ed-91b4-014921173900)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/11d299d1-532a-4ef4-a5a0-6a7bb93c1126)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/9085eab6-4091-4548-8c61-5fe875a19cc2)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/cc435d8b-e5c0-4bab-88b3-f66de29a48a3)

## [examples/color_matrix/sand_world.py](examples/color_matrix/sand_world.py)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/f2a61fae-5133-4e2c-8bf9-71e69c1d0948)

## [examples/lighting/light_source.py](examples/lighting/light_source.py)

![download (1)](https://github.com/jackparmer/VoxelWorld/assets/1865834/d86f3e6a-322a-4273-8260-fc41fb215eaf)

## [examples/color_matrix/jill_of_the_jungle.py](examples/color_matrix/jill_of_the_jungle.py)

![jill_of_the_jungle](https://github.com/jackparmer/VoxelWorld/assets/1865834/820494a5-452f-4f87-b6c7-bbe4abc3e65e)

## [examples/color_matrix/earth_tones.py](examples/color_matrix/earth_tones.py)

![earth_tones](https://github.com/jackparmer/VoxelWorld/assets/1865834/1cffc6bf-a07c-4804-86fa-783dae51b3b6)

## Mono-color themes

```py
from voxel_world import Volume

world = Volume.show_themes() # Jupyter notebook only
```
![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/ab7eca82-5b20-4b7e-bbae-a2e8350b4611)
