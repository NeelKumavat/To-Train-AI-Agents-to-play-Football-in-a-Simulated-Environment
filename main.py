import pygame
from Environment import Trainer

if __name__ == '__main__':
    trainer = Trainer(generations=1, match_duration=10, distance_weight=1.5, goal_weight=3)
    trainer.train()
    pygame.quit()