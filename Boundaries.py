import pygame

class Boundary:

    def __init__(self, x1, y1, x2, y2):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, Screen):

        pygame.draw.line(Screen, (0, 0, 0), (self.x1, self.y1), (self.x2, self.y2), 2)

boundaries = []

b1 = Boundary(0, 449, 100, 449)
b2 = Boundary(200, 449, 300, 449)
b3 = Boundary(0, 0, 0, 450)
b4 = Boundary(299, 0, 299, 450)
b5 = Boundary(0, 0, 100, 0)
b6 = Boundary(200, 0, 300, 0)

boundaries.append(b1)
boundaries.append(b2)
boundaries.append(b3)
boundaries.append(b4)
boundaries.append(b5)
boundaries.append(b6)

def getBoundaries():

    return boundaries