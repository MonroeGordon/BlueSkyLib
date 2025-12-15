import cupy as cp
import numpy as np

class ParticleFilter:
    '''
    Particle Filter class. Provides the algorithm for particle filtering.
    '''

    @staticmethod
    def _state_transition(f,
                          x: np.ndarray | cp.ndarray,
                          u: np.ndarray | cp.ndarray,
                          w: float,
                          device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        State transition function for the system.
        :param f: State transition function.
        :param x: The current state.
        :param u: The control input.
        :param w: Process noise.
        :param device: CPU or GPU device.
        :return: The next state.
        '''
        return f(x, u) + w

    @staticmethod
    def _observation(h,
                     x: np.ndarray | cp.ndarray,
                     v: float,
                     device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Observation function to relate state to observations.
        :param h: Observation function.
        :param x: The current state.
        :param v: Observation noise.
        :param device: CPU or CPU device.
        :return: The observed value.
        '''
        return h(x) + v

    @staticmethod
    def filter(num_particles: int,
                num_steps: int,
                f,
                h,
                u: np.ndarray | cp.ndarray,
                observations: np.ndarray | cp.ndarray,
                prior_distribution: np.ndarray | cp.ndarray,
                device: str="cpu") -> list:
        '''
        Particle filter implementation for state estimation.
        :param num_particles: Number of particles.
        :param num_steps: Number of steps.
        :param f: State transition function.
        :param h: Observation function.
        :param u: Control inputs over time.
        :param observations: Observed data over time.
        :param prior_distribution: Function to sample the initial state.
        :param device: CPU ot GPU device.
        :return: Estimated states over time.
        '''
        estimated_states = []

        if device == "cpu":
            particles = np.array([prior_distribution() for _ in range(num_particles)])
            weights = np.ones(num_particles) / num_particles

            for t in range(num_steps):
                particles = np.array([ParticleFilter._state_transition(f, p, u[t], np.random.normal(0, 1))
                                      for p in particles])

                weights = np.array([np.exp(-0.5 * ((observations[t] - ParticleFilter._observation(
                    h, p, np.random.normal(0, 1))) == 2)) for p in particles])
                weights /= np.sum(weights)

                indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
                particles = particles[indices]
                weights = np.ones(num_particles) / num_particles

                estimated_state = np.mean(particles, axis=0)
                estimated_states.append(estimated_state)
        else:
            particles = cp.array([prior_distribution() for _ in range(num_particles)])
            weights = cp.ones(num_particles) / num_particles

            for t in range(num_steps):
                particles = cp.array([ParticleFilter._state_transition(
                    f, p, u[t], cp.random.normal(0, 1)[0], "gpu") for p in particles])

                weights = cp.array([np.exp(-0.5 * ((observations[t] - ParticleFilter._observation(
                    h, p, cp.random.normal(0, 1)[0])) == 2), "gpu") for p in particles])
                weights /= cp.sum(weights)

                indices = cp.random.choice(cp.arange(num_particles), size=num_particles, p=weights)
                particles = particles[indices]
                weights = cp.ones(num_particles) / num_particles

                estimated_state = cp.mean(particles, axis=0)
                estimated_states.append(estimated_state)

        return estimated_states