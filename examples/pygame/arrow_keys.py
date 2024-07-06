import io
import pygame
import numpy as np
from pygame.locals import *
from voxel_world import Volume, Surface, Agent

SIZE = 48
SCALE = SIZE//16
WIN_WIDTH = 400*SCALE
WIN_HEIGHT = 300*SCALE

volume = Volume(Volume.purlin_matrix(SIZE)) # Voxel world
surf = Surface(volume) # World surface
agent = Agent(surf).cell(10, 10) # Surface agent with initial position @ 10, 10

def byte_stream(im):
    byte_stream = io.BytesIO()
    im.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    return byte_stream

pygame.init()
pygame.key.set_repeat(100, 100)
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
im = volume.add(agent) # Agent overlaid on Volume
pyg_img = pygame.image.load(byte_stream(im))
done = False
bg = (127,127,127)
mvmts = dict(up=(0,-1), down=(0,1), left=(-1,0), right=(1,0))

while not done:
    for event in pygame.event.get():
        screen.fill(bg)
        rect = pyg_img.get_rect()
        rect.center = WIN_WIDTH // 2, WIN_HEIGHT // 2
        screen.blit(pyg_img, rect)
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            key=pygame.key.name(event.key)
            if key in ['up', 'down', 'left', 'right']:
                x, y = np.nonzero(agent.mask)
                x += mvmts[key][0]
                y += mvmts[key][1]
                if x < surf.topology.shape[0] and y < surf.topology.shape[1]:
                    agent.cell(x, y)
                    im = volume.add(agent) # Agent overlaid on Volume
                    pyg_img = pygame.image.load(byte_stream(im))
                    screen.blit(pyg_img, rect)
        if event.type == pygame.KEYUP:
            key=pygame.key.name(event.key)
    pygame.display.update()