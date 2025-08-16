import pygame

class GoalPost:

    def __init__(self, x1, y1, x2, y2):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, Screen):

        pygame.draw.line(Screen, (255, 0, 0), (self.x1, self.y1), (self.x2, self.y2), 2)


goalPosts = []

g1 = GoalPost(100, 0, 200, 0)
g2 = GoalPost(100, 449, 200, 449)

goalPosts.append(g1)
goalPosts.append(g2)

def getGoalPosts():

    return goalPosts