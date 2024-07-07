# VoxelWorld

3d Numpy array in -> Voxel world out

[Demo on Py.Cafe ☕](https://py.cafe/jackparmer/voxel-worlds)

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

For physics simulation, computer vision, games, art, whatever

Inspo: https://github.com/wwwtyro/vixel

Features!
- Automatic GIF generation
- Numpy 3d ones array in -> Voxel world out
- Fast-ish (as fast as rendering on the CPU can be)
- Portable! Outputs simple image files
- Notebooks! Works well in the Jupyter notebook ecosystem
- Eye candy! [Ambient occlusion](https://en.wikipedia.org/wiki/Ambient_occlusion), ray tracing from Vixel, etc

***

# Examples

## Ray tracing + WebGL renderer

```py
from voxel_world import Volume, Vixel; vw = Volume(); vix = Vixel(vw); vix.html()
```

<img width="757" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/90826a0c-6d74-4956-acd1-fa230a79c9da">

## Animations

```py
from voxel_world import Volume, Surface, Agent, Sequence

volume = Volume(Volume.purlin_matrix(64));
surf = Surface(volume);
agents = [Agent(surf, mask) for mask in Sequence.snake(grid_size=64, num_steps=1000)];
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
<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/4a9c5f99-4ff2-441b-9086-fac3c4e7132a">

## Randomly generated worlds

[Live demo on Py.Cafe ☕](https://py.cafe/jackparmer/voxel-worlds)

```py
import random; import vnoise
import numpy as np
from voxel_world import Volume
noise = vnoise.Noise()

Volume(
    np.array([[[1 if noise.noise3(x / 10.0, y / 10.0, z / 10.0) > random.uniform(-0.2, 0.2) else 0 for z in range(16)] for y in range(16)] for x in range(16)], dtype=np.uint8),
    theme=random.choice(list(Volume.themes.keys())),
    resolution=10,
    viewing_angle=(random.randint(0, 90), random.randint(0, 90)),
    zoom=2.0,
    show_light_source=False,
    dark_bg=False
).render().show()
```
<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/80ad3ed5-15f2-427f-9608-72a46b07e932">

## [examples/color_matrix/sand_world.py](examples/color_matrix/sand_world.py)

<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/f2a61fae-5133-4e2c-8bf9-71e69c1d0948">

## [examples/lighting/light_source.py](examples/lighting/light_source.py)

<img width="800" alt="image" src="https://github.com/jackparmer/VoxelWorld/assets/1865834/d86f3e6a-322a-4273-8260-fc41fb215eaf">

