# VoxelWorld
Create delicious Voxel worlds in Python

## Install

```sh
git clone https://github.com/jackparmer/VoxelWorld.git
cd VoxelWorld
python3 -m pip install .
```

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

## Randomly generated worlds

```py
import random
from noise import pnoise3

display(IPImage(data=VoxelWorld.Animations.create_voxel_img(
    np.array([[[1 if pnoise3(x / 10.0, y / 10.0, z / 10.0) > random.uniform(-0.2, 0.2) else 0 for z in range(16)] for y in range(16)] for x in range(16)], dtype=np.uint8),
    random.choice(list(VoxelWorld.themes.keys())),
    resolution=10,
    viewing_angle=(random.randint(0, 90), random.randint(0, 90)),
    zoom=2.0,
    show_light_source=False,
    dark_bg=False
).getvalue()))
```

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/25bd612e-b8e9-42ed-91b4-014921173900)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/11d299d1-532a-4ef4-a5a0-6a7bb93c1126)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/9085eab6-4091-4548-8c61-5fe875a19cc2)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/cc435d8b-e5c0-4bab-88b3-f66de29a48a3)

## [examples/sandworld.py](examples/sand_world.py)

![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/f2a61fae-5133-4e2c-8bf9-71e69c1d0948)

## [examples/light_source.py](examples/light_source.py)

![download (1)](https://github.com/jackparmer/VoxelWorld/assets/1865834/d86f3e6a-322a-4273-8260-fc41fb215eaf)

## [examples/jill_of_the_jungle.py](examples/jill_of_the_jungle.py)

![jill_of_the_jungle](https://github.com/jackparmer/VoxelWorld/assets/1865834/820494a5-452f-4f87-b6c7-bbe4abc3e65e)

## [examples/earth_tones.py](examples/earth_tones.py)

![earth_tones](https://github.com/jackparmer/VoxelWorld/assets/1865834/1cffc6bf-a07c-4804-86fa-783dae51b3b6)

## Mono-color themes

```py
world = VoxelWorld.show_themes()
```
![image](https://github.com/jackparmer/VoxelWorld/assets/1865834/ab7eca82-5b20-4b7e-bbae-a2e8350b4611)
