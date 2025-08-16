import pygame
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import h5  # Import our new genome storage module
import h5py
from Boundaries import getBoundaries
from GoalPosts import getGoalPosts

# ---------------------------
#  Helper: Rotate Image
# ---------------------------
def rotate_center(Screen, image, angle, topLeft):
    rotated = pygame.transform.rotate(image, angle)
    new_rect = rotated.get_rect(center=image.get_rect(topleft=topLeft).center)
    return Screen.blit(rotated, new_rect.topleft)

# ---------------------------
#  Ray Intersection Helpers
# ---------------------------
def ray_segment_intersection(origin, direction, p1, p2):
    x1, y1 = origin
    dx, dy = direction
    x2, y2 = p1
    x3, y3 = p2
    seg_dx = x3 - x2
    seg_dy = y3 - y2
    denominator = dx * seg_dy - dy * seg_dx
    if abs(denominator) < 1e-6:
        return None
    t = ((x2 - x1) * seg_dy - (y2 - y1) * seg_dx) / denominator
    u = ((x2 - x1) * dy - (y2 - y1) * dx) / denominator
    if t >= 0 and 0 <= u <= 1:
        return t
    return None

def ray_circle_intersection(origin, direction, center, radius):
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    fx = ox - cx
    fy = oy - cy
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    t = None
    if t1 >= 0 and t2 >= 0:
        t = min(t1, t2)
    elif t1 >= 0:
        t = t1
    elif t2 >= 0:
        t = t2
    if t is not None and t >= 0:
        return t
    return None

# ---------------------------
#  Game Object Classes
# ---------------------------
class Football:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.footballImg = pygame.image.load('Images/Football.png')
        self.footballMask = pygame.mask.from_surface(self.footballImg)
        self.vx = 0  
        self.vy = 0  
        self.friction = 0.98
        self.restitution = 0.8
        self.size = 35

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction
        if self.x <= 0:
            self.x = 0
            self.vx = -self.vx * self.restitution
        if self.x + self.size >= 300:
            self.x = 300 - self.size
            self.vx = -self.vx * self.restitution
        if self.y <= 0:
            self.y = 0
            self.vy = -self.vy * self.restitution
        if self.y + self.size >= 450:
            self.y = 450 - self.size
            self.vy = -self.vy * self.restitution

    def draw(self, Screen):
        rotate_center(Screen, self.footballImg, 0, topLeft=(self.x, self.y))

class Player:
    def __init__(self, x, y, imagePath, angle, policy_params=None):
        self.x = x
        self.y = y
        self.playerImg = pygame.image.load(imagePath)
        self.playerMask = pygame.mask.from_surface(self.playerImg)
        self.vel = 5
        self.rot_vel = 5
        self.angle = angle
        self.radius = 17
        self.fitness = 0
        # Fitness tracking
        self.goals = 0
        self.touches = 0
        self.distance_sum = 0.0
        self.distance_count = 0
        # No neural network or policy_params used

    def reset_fitness_stats(self):
        self.fitness = 0
        self.goals = 0
        self.touches = 0
        self.distance_sum = 0.0
        self.distance_count = 0

    def collision(self, other_player):
        offset = (int(self.x - other_player.x), int(self.y - other_player.y))
        return self.playerMask.overlap(other_player.playerMask, offset)

    def resolve_collision(self, other_player):
        dx = self.x - other_player.x
        dy = self.y - other_player.y
        dist = math.hypot(dx, dy)
        
        if dist < self.radius + other_player.radius:  # If players are colliding
            overlap = self.radius + other_player.radius - dist
            angle = math.atan2(dy, dx)

            # Move the players apart by the amount of overlap
            move_x = math.cos(angle) * overlap / 2
            move_y = math.sin(angle) * overlap / 2

            # Move the players apart
            self.x += move_x
            self.y += move_y
            other_player.x -= move_x
            other_player.y -= move_y

    def boundary_check(self):
        if self.x <= 0: self.x = 0
        if self.x + 35 >= 300: self.x = 300 - 35
        if self.y <= 1: self.y = 1
        if self.y + 35 >= 450: self.y = 450 - 35

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rot_vel
        if right:
            self.angle -= self.rot_vel

    def move_forward(self):
        radians = math.radians(self.angle)
        ver = math.cos(radians) * self.vel
        hor = math.sin(radians) * self.vel
        self.x -= hor
        self.y -= ver

    def move_backward(self):
        radians = math.radians(self.angle)
        ver = math.cos(radians) * self.vel
        hor = math.sin(radians) * self.vel
        self.x += hor
        self.y += ver

    def draw(self, Screen):
        rotate_center(Screen, self.playerImg, self.angle, topLeft=(self.x, self.y))

    def decide_action(self, state=None):
        env = self.env
        is_red = (env.players[0] is self)
        ball_x = env.football.x + 17
        ball_y = env.football.y + 17
        player_x = self.x + 17
        player_y = self.y + 17
        dx = ball_x - player_x
        dy = ball_y - player_y
        dist = math.hypot(dx, dy)
        margin = 20
        # Wall avoidance
        if self.x < margin:
            self.x += self.vel
        elif self.x + 35 > env.width - margin:
            self.x -= self.vel
        if self.y < margin:
            self.y += self.vel
        elif self.y + 35 > env.height - margin:
            self.y -= self.vel
        # Repulsion if too close to the other player (to avoid ball stuck)
        other = env.players[1] if is_red else env.players[0]
        other_x = other.x + 17
        other_y = other.y + 17
        dist_to_other = math.hypot(player_x - other_x, player_y - other_y)
        if dist_to_other < 38:
            repel_dx = player_x - other_x
            repel_dy = player_y - other_y
            repel_dist = math.hypot(repel_dx, repel_dy)
            if repel_dist > 1e-2:
                self.x += (repel_dx / repel_dist) * self.vel * 0.5
                self.y += (repel_dy / repel_dist) * self.vel * 0.5
        # --- RANDOM ACTION LOGIC FOR RED PLAYER ---
        if is_red:
            # RED: chase the ball and try to score
            kick_distance = 25  # Allow kicking from farther away
            if dist > 40:
                step = self.vel
                move_x = (dx / dist) * step
                move_y = (dy / dist) * step
                self.x += move_x
                self.y += move_y
                self.angle = ((math.degrees(math.atan2(dy, dx)) - 90) + 360) % 45
            elif dist <= 40 and dist > kick_distance:
                step = self.vel
                move_x = (dx / dist) * step
                move_y = (dy / dist) * step
                self.x += move_x
                self.y += move_y
                self.angle = ((math.degrees(math.atan2(dy, dx)) - 90) + 360) % 45
            else:
                goal_x = 150  # center x of field
                goal_y = 0    # top goal y
                to_goal_x = goal_x - player_x
                to_goal_y = goal_y - player_y
                to_goal_dist = math.hypot(to_goal_x, to_goal_y)
                if to_goal_dist > 0:
                    move_x = (to_goal_x / to_goal_dist) * self.vel
                    move_y = (to_goal_y / to_goal_dist) * self.vel
                    self.x += move_x
                    self.y += move_y
                    self.angle = ((math.degrees(math.atan2(to_goal_y, to_goal_x)) - 90) + 360) % 360
        else:
            # BLUE: follow the ball in both x and y directions
            move_step = self.vel
            if abs(ball_x - player_x) > 2:
                self.x += move_step if ball_x > player_x else -move_step
            if abs(ball_y - player_y) > 2:
                self.y += move_step if ball_y > player_y else -move_step
            # Clamp to field boundaries
            self.x = max(0, min(self.x, env.width - 35))
            self.y = max(0, min(self.y, env.height - 35))
            # Face the field
            self.angle = 180
            # If close enough to the ball, shoot in a random direction between -60 and 60 degrees
            dist_to_ball = math.hypot(ball_x - (self.x + 17), ball_y - (self.y + 17))
            if dist_to_ball < 40:
                random_angle = random.uniform(-60, 60)
                self.angle = 180 + random_angle
        return None

class SimpleNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=4):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
#  Environment Class
# ---------------------------
class FootbalEnv:
    def __init__(self, agents=None):
        pygame.init()
        self.width = 300
        self.height = 450
        self.fps = 120
        self.boundaries = getBoundaries()
        self.goalPosts = getGoalPosts()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("RL Football with Genetic Algorithm")
        self.back_image = pygame.image.load('Images/Field.png')
        self.football = Football(137, 212)
        self.score = {"RED": 0, "BLUE": 0}
        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        if agents is None:
            self.players = [
                Player(130, 380, 'Images/Player1.png', angle=0),
                Player(130, 40, 'Images/Player2.png', angle=180)
            ]
            self.players[1].vel = 5
        else:
            self.players = [
                Player(130, 380, 'Images/Player1.png', angle=0, policy_params=agents[0]),
                Player(130, 40, 'Images/Player2.png', angle=180, policy_params=agents[1])
            ]
            self.players[1].vel = 5
        # Attach env reference to each player for ball chasing logic
        for player in self.players:
            player.env = self
        self.num_rays = 64
        self.fov = 360
        # Fitness weights
        self.goal_weight = 5.0
        self.touch_weight = 1.0
        self.distance_weight = 1.0

        # Neural network creation and genome.h5 saving (not used for player logic)
        self.save_genomes_if_needed()

    def save_genomes_if_needed(self):
        import os
        genome_dir = 'genome'
        if not os.path.exists(genome_dir):
            os.makedirs(genome_dir)
        red_path = os.path.join(genome_dir, 'red.h5')
        blue_path = os.path.join(genome_dir, 'blue.h5')
        if not os.path.exists(red_path):
            net_red = SimpleNet()
            with h5py.File(red_path, 'w') as f:
                f.create_dataset('weights', data=np.concatenate([p.data.cpu().numpy().flatten() for p in net_red.parameters()]))
        if not os.path.exists(blue_path):
            net_blue = SimpleNet()
            with h5py.File(blue_path, 'w') as f:
                f.create_dataset('weights', data=np.concatenate([p.data.cpu().numpy().flatten() for p in net_blue.parameters()]))
        # Removed print statements about file creation

    def reset(self, soft=True):
        # Ball spawns at a random location each reset
        ball_x = random.randint(0, self.width - 35)
        ball_y = random.randint(0, self.height - 35)
        self.football = Football(ball_x, ball_y)
        self.players[0].x, self.players[0].y = 130, 380
        self.players[1].x, self.players[1].y = 130, 40
        for player in self.players:
            player.reset_fitness_stats()
        if not soft: self.score = {"RED": 0, "BLUE": 0}

    def check_goal(self):
        scored = False
        ball_center_x = self.football.x + 17
        if self.football.y <= 0 and 100 <= ball_center_x <= 200:
            self.players[0].goals += 1  # Red scores
            self.players[0].fitness = self.compute_fitness(self.players[0])
            self.players[1].fitness = self.compute_fitness(self.players[1])
            self.score['RED'] += 1
            scored = True
        elif self.football.y + self.football.size >= self.height and 100 <= ball_center_x <= 200:
            self.players[1].goals += 1  # Blue scores
            self.players[0].fitness = self.compute_fitness(self.players[0])
            self.players[1].fitness = self.compute_fitness(self.players[1])
            self.score['BLUE'] += 1
            scored = True
        if scored:
            # Reset stats for both players after fitness is updated
            for player in self.players:
                player.reset_fitness_stats()
            self.reset()

    def compute_fitness(self, player):
        avg_dist = (player.distance_sum / player.distance_count) if player.distance_count > 0 else 0.0
        # Fitness is always positive and higher is better
        # Add a constant to denominator to avoid division by zero
        return (
            player.goals * self.goal_weight +
            player.touches * self.touch_weight +
            self.distance_weight / (avg_dist + 1)
        )

    def update(self):
        self.football.update()
        # Update distance to ball for each player
        for player in self.players:
            ball_x = self.football.x + 17
            ball_y = self.football.y + 17
            player_x = player.x + 17
            player_y = player.y + 17
            dist = math.hypot(ball_x - player_x, ball_y - player_y)
            player.distance_sum += dist
            player.distance_count += 1
        self.check_goal()

    def handle_collisions(self):
        # Check for collision between each player and the ball.
        for idx, player in enumerate(self.players):
            player_center = (player.x + 17, player.y + 17)
            ball_center = (self.football.x + 17, self.football.y + 17)
            dx = player_center[0] - ball_center[0]
            dy = player_center[1] - ball_center[1]
            dist = math.hypot(dx, dy)
            if dist < (player.radius + 17):  # collision threshold
                # Count touch
                player.touches += 1
                # Increase kick power for blue player
                if player is self.players[1]:
                    force = 25  # Blue player kick power increased
                else:
                    force = 15  # Red player default kick power
                self.football.vx = math.sin(math.radians(player.angle)) * force
                self.football.vy = -math.cos(math.radians(player.angle)) * force
                # --- Ball stuck fix: if ball is too close for too long, push it behind player ---
                stuck_margin = 5
                if dist < (player.radius + 17) - stuck_margin:
                    angle_rad = math.radians(player.angle)
                    behind_x = player_center[0] - math.sin(angle_rad) * (player.radius + 17 + 2)
                    behind_y = player_center[1] + math.cos(angle_rad) * (player.radius + 17 + 2)
                    self.football.x = behind_x - 17
                    self.football.y = behind_y - 17
                    self.football.vx = -math.sin(angle_rad) * 5
                    self.football.vy = math.cos(angle_rad) * 5

        # Check for collision between players
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):  # Ensure each pair is checked only once
                player1 = self.players[i]
                player2 = self.players[j]
                if player1.collision(player2):
                    player1.resolve_collision(player2)  # Resolve the collision

    def render(self, show_rays=False):
        self.screen.fill((154, 247, 100))
        self.screen.blit(self.back_image, (0, 0))
        for obstacle in self.boundaries:
            obstacle.draw(self.screen)
        for goalPost in self.goalPosts:
            goalPost.draw(self.screen)
        self.football.draw(self.screen)
        # Always show rays
        if show_rays:
            for player in self.players:
                self.cast_and_draw_rays(player)
        for player in self.players:
            player.boundary_check()
            player.draw(self.screen)

        # Update fitness live for both players
        self.players[0].fitness = self.compute_fitness(self.players[0])
        self.players[1].fitness = self.compute_fitness(self.players[1])

        # Display fitness and score at top and bottom center (previous positions)
        blue_text = self.font.render(f"BLUE: {self.score['BLUE']}  Fitness: {self.players[1].fitness:.1f}", True, pygame.Color('blue'))
        self.screen.blit(blue_text, dest=(90, 10))  # Top center
        red_text = self.font.render(f"RED: {self.score['RED']}  Fitness: {self.players[0].fitness:.1f}", True, pygame.Color('red'))
        self.screen.blit(red_text, dest=(90, 430))  # Bottom center
        
        pygame.display.update()

    def step(self, actions):
        for idx, action in enumerate(actions):
            if action == 'move_forward':
                self.players[idx].move_forward()
            elif action == 'rotate_left':
                self.players[idx].rotate(left=True)
            elif action == 'rotate_right':
                self.players[idx].rotate(right=True)
        self.handle_collisions()
        self.update()

    def calculate_fitness(self):
        # Fitness = -avg_dist_to_ball + 10*goals + 2*ball_touches
        fitness = [0, 0]
        for idx, player in enumerate(self.players):
            # Average distance to ball (sampled at each step)
            if not hasattr(self, 'distance_history'):
                self.distance_history = [[], []]
            player_center = (player.x + 17, player.y + 17)
            ball_center = (self.football.x + 17, self.football.y + 17)
            dist = math.hypot(player_center[0] - ball_center[0], player_center[1] - ball_center[1])
            self.distance_history[idx].append(dist)
            avg_dist = np.mean(self.distance_history[idx]) if self.distance_history[idx] else 0
            goals = self.score['RED'] if idx == 0 else self.score['BLUE']
            fitness[idx] = -avg_dist + 10 * goals + 2 * self.touches[idx]
        return fitness

    def get_ray_state(self, player):
        state = []
        origin = (player.x + 17, player.y + 17)
        start_angle = player.angle - self.fov / 2
        angle_step = self.fov / (self.num_rays - 1)
        max_distance = 300.0
        for i in range(self.num_rays):
            ray_angle = start_angle + i * angle_step
            ray_rad = math.radians(ray_angle)
            direction = (math.sin(ray_rad), -math.cos(ray_rad))
            min_t = max_distance
            for obstacle in self.boundaries + self.goalPosts:
                p1 = (obstacle.x1, obstacle.y1)
                p2 = (obstacle.x2, obstacle.y2)
                t = ray_segment_intersection(origin, direction, p1, p2)
                if t is not None and t < min_t:
                    min_t = t
            ball_center = (self.football.x + 17, self.football.y + 17)
            t = ray_circle_intersection(origin, direction, ball_center, 17)
            if t is not None and t < min_t:
                min_t = t
            state.append(min_t / max_distance)
        return state

    def cast_and_draw_rays(self, player):
        origin = (player.x + 17, player.y + 17)
        start_angle = player.angle - self.fov / 2
        angle_step = self.fov / (self.num_rays - 1)
        max_distance = 300
        # Color logic: blue player (self.players[1]) gets red rays, red player (self.players[0]) gets blue rays
        if player is self.players[1]:
            ray_color = (0, 0, 0)  # Red for blue player
        else:
            ray_color = (0, 0, 0)  # Blue for red player
        for i in range(self.num_rays):
            ray_angle = start_angle + i * angle_step
            ray_rad = math.radians(ray_angle)
            direction = (math.sin(ray_rad), -math.cos(ray_rad))
            min_t = max_distance
            intersection_point = (origin[0] + direction[0] * max_distance,
                                  origin[1] + direction[1] * max_distance)
            for obstacle in self.boundaries + self.goalPosts:
                p1 = (obstacle.x1, obstacle.y1)
                p2 = (obstacle.x2, obstacle.y2)
                t = ray_segment_intersection(origin, direction, p1, p2)
                if t is not None and t < min_t:
                    min_t = t
                    intersection_point = (origin[0] + direction[0] * t,
                                          origin[1] + direction[1] * t)
            ball_center = (self.football.x + 17, self.football.y + 17)
            t = ray_circle_intersection(origin, direction, ball_center, 17)
            if t is not None and t < min_t:
                min_t = t
                intersection_point = (origin[0] + direction[0] * t,
                                      origin[1] + direction[1] * t)
            pygame.draw.line(self.screen, ray_color, origin, intersection_point, 1)

# ---------------------------
#  Genetic Algorithm Trainer
# ---------------------------


class Trainer:

    def __init__(self, generations=1, match_duration=10, distance_weight=1.0, goal_weight=5.0):
        self.generations = generations
        self.match_duration = float("inf")  # Always running
        self.distance_weight = distance_weight
        self.goal_weight = goal_weight
        self.fps = 50

    def run_match(self, render=False):
        env = FootbalEnv()
        env.reset(soft=False)
        clock = pygame.time.Clock()
        start_time = time.time()
        while time.time() - start_time < self.match_duration:
            for player in env.players:
                player.decide_action()
            env.handle_collisions()
            env.update()
            if render:
                env.render(show_rays=True)
                clock.tick(self.fps)
        return self.calculate_fitness(env)

    def calculate_fitness(self, env):
        return env.calculate_fitness()

    def train(self):
        env = FootbalEnv()
        env.reset(soft=False)
        clock = pygame.time.Clock()
        while True:
            match_start = time.time()
            while time.time() - match_start < self.match_duration:
                for player in env.players:
                    player.decide_action()
                env.handle_collisions()
                env.update()
                env.render(show_rays=True)
                clock.tick(self.fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            env.reset(soft=False)


