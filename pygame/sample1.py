#!/usr/bin/env python
# -*0 coding: utf-8 -*-

import pygame
from time import sleep
from pdb import *

pygame.init()

size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
print "Framebuffer size: %d x %d" % (size[0], size[1])
#screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
screen = pygame.display.set_mode((size[0]/2,size[1]/2), pygame.RESIZABLE)
#keifont = pygame.font.Font("keifont.ttf", 80)
for i in range(0,255,15):
    for j in range(0,255,15):
        #title = keifont.render(u"なかよしデジカメ", True, (180,0,0))
        screen.fill((i,j,int((i*j)/255)))
    
        #screen.blit("abc", (20,150))
        pygame.display.update()
        sleep(0.03)
