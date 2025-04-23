from environment import SimulationEnvironment
from agent import DQNAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

if __name__ == "__main__":

    env = SimulationEnvironment()

    # Define input dimension and output dimension based on state and action spaces
    input_dim = len(env.reset())  # Length of the state representation
    output_dim = env.action_space.shape[0]  # Dimensionality of the action space

    agent = DQNAgent(env, input_dim, output_dim)

    # Train the agent
    rewards = agent.train(num_episodes=100)

    # Plot the rewards over episodes
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

    final_env = SimulationEnvironment()
    state = final_env.reset()
    states = [final_env.get_entity_positions()]

    done = False
    while not done:
        # Use agent's method to compute action from flat state
        action = agent.act(state)
        next_state, reward, done = final_env.step(action)
        states.append(final_env.get_entity_positions())
        state = next_state

    # Extract relevant information for visualization
    time_stamps = [s["time"] for s in states]
    tx_states = [s["transmitter_state"] for s in states]  # [rotation, azimuth, elevation, x, y, z]
    drone_states = [s["drone_state"] for s in states]     # each: [x, y, z, vx, vy, vz, drone_id]
    target_ids = [s["target"].drone_id for s in states]

    # Set up the plot for visualization
    fig, ax = plt.subplots()
    scat_tx = ax.plot([], [], 'ro', label='Transmitter')[0]
    scat_drones = ax.plot([], [], 'b^', label='Drones')[0]
    beam_line, = ax.plot([], [], color='yellow', linewidth=2, label='Beam Direction')
    timestamp_text = ax.text(20, 28, '', fontsize=11, color='gray')
    target_text = ax.text(0.98, 0.02, '', fontsize=11, color='gray', ha='right', va='bottom', transform=ax.transAxes)

    ax.set_xlim(0, final_env.space_bounds[0])
    ax.set_ylim(0, final_env.space_bounds[1])
    ax.set_title('Simulation Environment Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    drone_labels = []

    def init():
        scat_tx.set_data([], [])
        scat_drones.set_data([], [])
        beam_line.set_data([], [])
        timestamp_text.set_text('0.00s')
        target_text.set_text("Target: -")

        for label in drone_labels:
            label.remove()
        drone_labels.clear()

        return scat_tx, scat_drones, beam_line, timestamp_text, target_text

    def update(frame):
        # Parse current frame state
        tx_state = tx_states[frame]
        tx_rotation = tx_state[0]
        tx_azimuth = tx_state[1]
        tx_x, tx_y = tx_state[3], tx_state[4]

        scat_tx.set_data([tx_x], [tx_y])
        timestamp_text.set_text(f"{time_stamps[frame]:.2f}s")

        # Parse drone positions and IDs
        drones = drone_states[frame]
        if drones:
            drones = np.array(drones)
            drone_xy = drones[:, :2]
            drone_ids = drones[:, -1].astype(int)

            scat_drones.set_data(drone_xy[:, 0], drone_xy[:, 1])

            # Remove old labels
            for label in drone_labels:
                label.remove()
            drone_labels.clear()

            # Add new labels for current frame
            for (x, y), drone_id in zip(drone_xy, drone_ids):
                label = ax.text(x, y + 10, f"{drone_id}", fontsize=8, color='black', ha='center')
                drone_labels.append(label)
        else:
            scat_drones.set_data([], [])
            for label in drone_labels:
                label.remove()
            drone_labels.clear()

        # Beam direction calculation
        beam_length = 50
        beam_angle = tx_rotation + tx_azimuth
        beam_dx = beam_length * np.cos(beam_angle)
        beam_dy = beam_length * np.sin(beam_angle)
        beam_line.set_data([tx_x, tx_x + beam_dx], [tx_y, tx_y + beam_dy])

        # Target drone display
        target_text.set_text(f"Target: {target_ids[frame]}")

        return scat_tx, scat_drones, beam_line, timestamp_text, target_text, *drone_labels

    ani = animation.FuncAnimation(
        fig, update, frames=len(states), init_func=init, interval=50, blit=False
    )
    ani.save("simulation_output.gif", writer='pillow', fps=10)
    plt.show()
