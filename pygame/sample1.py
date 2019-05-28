#!/usr/bin/python
# -*0 coding: utf-8 -*-

import pygame
from time import sleep
from pdb import *

pygame.init()

size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
print "Framebuffer size: %d x %d" % (size[0], size[1])
#screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
screen = pygame.display.set_mode(size, pygame.RESIZABLE)
#keifont = pygame.font.Font("keifont.ttf", 80)
while True:
    #title = keifont.render(u"なかよしデジカメ", True, (180,0,0))
    screen.fill((255,230,0))
    
    #screen.blit("abc", (20,150))
    pygame.display.update()
    sleep(5)
    break
