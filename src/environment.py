import gym
import numpy as np
from drone import Drone
from transmitter import Transmitter
import random

class SimulationEnvironment:
    def __init__(self, num_drones=5, sim_duration=60, step_time=0.1):
        self.num_drones = num_drones
        self.sim_duration = sim_duration
        self.step_time = step_time
        self.total_steps = int(sim_duration / step_time)
        self.current_step = 0

        self.space_bounds = np.array([1000, 1000, 100])
        self.transmitter = Transmitter(position=np.array([500, 500, 50]))
        self.drones = [Drone(self.space_bounds, self.step_time, drone_id=i) for i in range(num_drones)]

        self.target = random.choice(self.drones)
        self._next_target_change = self._schedule_target_change()

        # Define the action space as a Box space with 3 dimensions:
        # rotation_velocity, beam_azimuth, beam_elevation
        # Bounds for each action:
        # rotation_velocity: [-π, π]
        # beam_azimuth: [-π/3, π/3]
        # beam_elevation: [-π/3, π/3]
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi/3, -np.pi/3]),
            high=np.array([np.pi, np.pi/3, np.pi/3]),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.drones = [Drone(self.space_bounds, self.step_time, drone_id=i) for i in range(self.num_drones)]
        self.transmitter.reset()
        self.target = random.choice(self.drones)
        self._next_target_change = self._schedule_target_change()
        return self._get_state()

    def _schedule_target_change(self):
        return self.current_step + int(random.uniform(3, 30) / self.step_time)

    def step(self, action):
        self.current_step += 1

        # Possibly update the target drone
        if self.current_step >= self._next_target_change:
            self.target = random.choice(self.drones)
            self._next_target_change = self._schedule_target_change()

        # Apply the action to the transmitter
        self.transmitter.apply_action(action)

        # Move all drones
        for drone in self.drones:
            drone.update(self.step_time)
            drone.avoid_bounds()

        # Calculate reward from the target drone only
        reward = self.calculate_rsrp()

        done = self.current_step >= self.total_steps
        next_state = self._get_state()

        return next_state, reward, done

    def calculate_rsrp(self):
        beam_dir = self.transmitter.get_beam_direction()
        beam_dir = beam_dir / np.linalg.norm(beam_dir)  # Ensure unit vector

        tx_pos = self.transmitter.position
        target_pos = self.target.position
        to_target = target_pos - tx_pos

        # Compute perpendicular distance from target to beam vector
        projection = np.dot(to_target, beam_dir)
        closest_point_on_beam = tx_pos + projection * beam_dir
        distance = np.linalg.norm(target_pos - closest_point_on_beam)

        # Exponential decay and beam strength
        power = (np.exp(-distance / 1.5) * self.transmitter.beam_strength)*100
        return power

    def _get_state(self):
        drone_states = [drone.get_state() for drone in self.drones]
        transmitter_state = self.transmitter.get_state()
        return np.concatenate(drone_states + [transmitter_state])

    def get_entity_positions(self):
        return {
            "time": self.current_step * self.step_time,
            "transmitter_state": self.transmitter.get_state(),
            "drone_state": [drone.get_state() for drone in self.drones],
            "target": self.target
        }