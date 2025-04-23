import numpy as np

class Transmitter:
    def __init__(self, position=np.array([0, 0, 0]), max_rotation_speed=np.pi * 2):
        self.position = position
        self.rotation = 0.0
        self.beam_azimuth = 0.0
        self.beam_elevation = 0.0
        self.max_rotation_speed = max_rotation_speed
        self.rotation_velocity = 0.0

        self.max_steering_angle = np.pi / 3  # 60 degrees

    def reset(self):
        self.rotation = 0.0
        self.beam_azimuth = 0.0
        self.beam_elevation = 0.0
        self.rotation_velocity = 0.0

    def apply_action(self, action):
        # Expect action as numpy array: [rotation_velocity, beam_azimuth, beam_elevation]
        self.rotation_velocity = np.clip(
            action[0],
            -self.max_rotation_speed,
            self.max_rotation_speed
        )
        self.beam_azimuth = np.clip(
            action[1],
            -self.max_steering_angle,
            self.max_steering_angle
        )
        self.beam_elevation = np.clip(
            action[2],
            -self.max_steering_angle,
            self.max_steering_angle
        )

    def update(self, step_time):
        self.rotation += self.rotation_velocity * step_time
        self.rotation = self.rotation % (2 * np.pi)

    def get_beam_direction(self):
        total_azimuth = self.rotation + self.beam_azimuth
        total_elevation = self.beam_elevation

        dx = np.cos(total_elevation) * np.cos(total_azimuth)
        dy = np.cos(total_elevation) * np.sin(total_azimuth)
        dz = np.sin(total_elevation)

        return np.array([dx, dy, dz])

    def _compute_main_lobe_strength(self):
        max_deflection = self.max_steering_angle
        az_factor = abs(self.beam_azimuth) / max_deflection
        el_factor = abs(self.beam_elevation) / max_deflection
        avg_deflection = (az_factor + el_factor) / 2
        return 1.0 - avg_deflection * 0.2  # Linearly scale from 1.0 to 0.8

    @property
    def beam_strength(self):
        return self._compute_main_lobe_strength()

    def get_state(self):
        return np.array([
            self.rotation,
            self.beam_azimuth,
            self.beam_elevation,
            *self.position
        ])
