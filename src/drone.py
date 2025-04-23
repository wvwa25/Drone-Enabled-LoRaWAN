import numpy as np
import random

class Drone:

    def __init__(self, bounds, step_time, drone_id=0):
        self.drone_id = drone_id  # unique ID for this drone
        self.bounds = bounds
        self.step_time = step_time
        self.position = np.random.uniform([0, 0, 0], bounds)
        self.velocity = np.zeros(3)
        self.acceleration = 9.26  # m/s^2
        self.max_speed = 13.9     # m/s
        self.state_timer = 0
        self.moving = True
        self._new_state_duration()
        self._random_direction()

    def _new_state_duration(self):
        self.state_timer = int(random.uniform(3, 30) / self.step_time)  # 3 to 30 seconds

    def avoid_bounds(self):
        margin = 5.0  # meters
        for i in range(3):  # x, y, z axes
            # Close to the lower bound and heading toward it
            if self.position[i] < margin and self.direction[i] < 0:
                self.direction[i] *= -1
            # Close to the upper bound and heading toward it
            elif self.position[i] > self.bounds[i] - margin and self.direction[i] > 0:
                self.direction[i] *= -1

        # Normalize direction after adjustment
        self.direction = self.direction / np.linalg.norm(self.direction)

    def _random_direction(self):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])

    def update(self, dt):
        if self.state_timer <= 0:
            self.moving = not self.moving
            self._new_state_duration()
            if self.moving:
                self._random_direction()

        if self.moving:
            speed = np.linalg.norm(self.velocity)
            if speed < self.max_speed:
                self.velocity += self.direction * self.acceleration * dt
                speed = min(np.linalg.norm(self.velocity), self.max_speed)
                self.velocity = self.direction * speed
            self.position += self.velocity * dt
            self.position = np.clip(self.position, [0, 0, 0], self.bounds)
        else:
            self.velocity = np.zeros(3)

        self.state_timer -= 1

    def get_state(self):
        return np.concatenate((self.position, self.velocity, np.array([self.drone_id])))
