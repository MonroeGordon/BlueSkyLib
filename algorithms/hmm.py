import cupy as cp
import numpy as np

class HMM:
    '''
    Hidden Markov Model (HMM) class. Provides algorithms for using a hidden markov model to calculate sequence
    probability, find the most likely hidden states, and estimate the next parameters of the hidden Markov model. This
    also provides algorithms for Markov chains.
    '''

    @staticmethod
    def estimate(o: np.ndarray | cp.ndarray,
                 a: np.ndarray | cp.ndarray,
                 b: np.ndarray | cp.ndarray,
                 pi: np.ndarray | cp.ndarray,
                 max_iter: int=100,
                 device: str = "cpu") -> tuple:
        '''
        Estimate parameters of hidden Markov model using the Baum-Welch algorithm.
        :param o: Observable symbol sequence (1-dimensional ndarray).
        :param a: Transition probability matrix (2-dimensional ndarray).
        :param b: Emission probability matrix (2-dimensional ndarray).
        :param pi: Initial state distribution (1-dimensional ndarray).
        :param max_iter: Maximum number of iterations for convergence.
        :param device: CPU or GPU device.
        :return: Updated a, b, and pi arrays.
        '''
        n = a.shape[0]
        t = len(o)

        if device == 'cpu':
            no = o
            na = a
            nb = b
            npi = pi

            if isinstance(no, cp.ndarray):
                no = cp.asnumpy(no)

            if isinstance(na, cp.ndarray):
                na = cp.asnumpy(na)

            if isinstance(nb, cp.ndarray):
                nb = cp.asnumpy(nb)

            if isinstance(npi, cp.ndarray):
                npi = cp.asnumpy(npi)

            for _ in range(max_iter):
                alpha = np.zeros((t, n))
                beta = np.zeros((t, n))
                xi = np.zeros((t - 1, n, n))
                gamma = np.zeros((t, n))
    
                for i in range(n):
                    alpha[0, i] = npi[i] * nb[i, no[0]]
                    
                for i in range(1, t):
                    for j in range(n):
                        alpha[i, j] = np.sum(alpha[i - 1] * na[:, j]) * nb[j, no[i]]
                        
                for i in range(n):
                    beta[t - 1, i] = 1
                    
                for i in range(t - 2, -1, -1):
                    for j in range(n):
                        beta[i, j] = np.sum(na[j] * nb[:, no[i + 1]] * beta[i + 1])
                        
                for i in range(t - 1):
                    denom = np.sum(alpha[i, :][:, None] * na * nb[:, o[i + 1]] * beta[i + 1, :])
                    
                    for j in range(n):
                        for k in range(n):
                            xi[i, j, k] = (alpha[i, j] * na[j, k] * nb[k, o[i + 1]] * beta[i + 1, k]) / denom
                            
                        gamma[i, j] = np.sum(xi[i, j, :])
                        
                for i in range(n):
                    gamma[t - 1, i] = alpha[t - 1, i] / np.sum(alpha[t - 1, :])
                    
                for i in range(n):
                    for j in range(n):
                        na[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
                        
                for i in range(n):
                    for j in range(nb.shape[1]):
                        nb[i, j] = np.sum(gamma[o == j, i]) / np.sum(gamma[:, i])
                        
                npi = gamma[0, :]
            
            return na, nb, npi
        else:
            co = o
            ca = a
            cb = b
            cpi = pi

            if isinstance(co, np.ndarray):
                co = cp.asarray(co)

            if isinstance(ca, np.ndarray):
                ca = cp.asarray(ca)

            if isinstance(cb, np.ndarray):
                cb = cp.asarray(cb)

            if isinstance(cpi, np.ndarray):
                cpi = cp.asarray(cpi)

            for _ in range(max_iter):
                alpha = np.zeros((t, n))
                beta = np.zeros((t, n))
                xi = np.zeros((t - 1, n, n))
                gamma = np.zeros((t, n))

                for i in range(n):
                    alpha[0, i] = cpi[i] * cb[i, co[0]]

                for i in range(1, t):
                    for j in range(n):
                        alpha[i, j] = np.sum(alpha[i - 1] * ca[:, j]) * cb[j, co[i]]

                for i in range(n):
                    beta[t - 1, i] = 1

                for i in range(t - 2, -1, -1):
                    for j in range(n):
                        beta[i, j] = np.sum(ca[j] * cb[:, co[i + 1]] * beta[i + 1])

                for i in range(t - 1):
                    decom = np.sum(alpha[i, :][:, None] * ca * cb[:, o[i + 1]] * beta[i + 1, :])

                    for j in range(n):
                        for k in range(n):
                            xi[i, j, k] = (alpha[i, j] * ca[j, k] * cb[k, o[i + 1]] * beta[i + 1, k]) / decom

                        gamma[i, j] = np.sum(xi[i, j, :])

                for i in range(n):
                    gamma[t - 1, i] = alpha[t - 1, i] / np.sum(alpha[t - 1, :])

                for i in range(n):
                    for j in range(n):
                        ca[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

                for i in range(n):
                    for j in range(cb.shape[1]):
                        cb[i, j] = np.sum(gamma[o == j, i]) / np.sum(gamma[:, i])

                cpi = gamma[0, :]

            return ca, cb, cpi

    @staticmethod
    def likely_states(o: np.ndarray | cp.ndarray,
                      a: np.ndarray | cp.ndarray,
                      b: np.ndarray | cp.ndarray,
                      pi: np.ndarray | cp.ndarray,
                      device: str = "cpu") -> np.ndarray | cp.ndarray:
        '''
        Finds the most likely sequence of hidden states using the Viterbi algorithm.
        :param o: Observable symbol sequence (1-dimensional ndarray).
        :param a: Transition probability matrix (2-dimensional ndarray).
        :param b: Emission probability matrix (2-dimensional ndarray).
        :param pi: Initial state distribution (1-dimensional ndarray).
        :param device: CPU or GPU device.
        :return: Most likely sequence of hidden states.
        '''
        n = a.shape[0]
        t = len(o)

        if device == 'cpu':
            no = o
            na = a
            nb = b
            npi = pi

            if isinstance(no, cp.ndarray):
                no = cp.asnumpy(no)

            if isinstance(na, cp.ndarray):
                na = cp.asnumpy(na)

            if isinstance(nb, cp.ndarray):
                nb = cp.asnumpy(nb)

            if isinstance(npi, cp.ndarray):
                npi = cp.asnumpy(npi)

            delta = np.zeros((t, n))
            psi = np.zeros((t, n), dtype=int)

            for i in range(n):
                delta[0, i] = npi[i] * nb[i, o[0]]

            for i in range(1, t):
                for j in range(n):
                    delta[i, j] = np.max(delta[i - 1] * na[:, j]) * nb[j, no[i]]
                    psi[i, j] = np.argmax(delta[i - 1] * na[:, j])

            x = np.zeros(t, dtype=int)
            x[t - 1] = np.argmax(delta[t - 1])

            for i in range(t - 2, -1, -1):
                x[i] = psi[i + 1, x[i + 1]]
        else:
            co = o
            ca = a
            cb = b
            cpi = pi

            if isinstance(co, np.ndarray):
                co = cp.asarray(co)

            if isinstance(ca, np.ndarray):
                ca = cp.asarray(ca)

            if isinstance(cb, np.ndarray):
                cb = cp.asarray(cb)

            if isinstance(cpi, np.ndarray):
                cpi = cp.asarray(cpi)

            delta = cp.zeros((t, n))
            psi = cp.zeros((t, n), dtype=int)

            for i in range(n):
                delta[0, i] = cpi[i] * cb[i, co[0]]

            for i in range(1, t):
                for j in range(n):
                    delta[i, j] = cp.max(delta[i - 1] * ca[:, j]) * cb[j, o[i]]
                    psi[i, j] = cp.argmax(delta[i - 1] * ca[:, j])

            x = cp.zeros(t, dtype=int)
            x[t - 1] = cp.argmax(delta[t - 1])

            for i in range(t - 2, -1, -1):
                x[i] = psi[i + 1, x[i + 1]]

        return x

    @staticmethod
    def sample(create_target_dist,
               create_proposal_dist,
               initial_state: int,
               iterations: int,
               device: str="cpu") -> list:
        '''
        Sample from a target distribution using the Metropolis-Hastings algorithm.
        :param create_target_dist: Function to create unnormalized target distributions.
        :param create_proposal_dist: Function to create proposal distributions (next state).
        :param initial_state: Initial state of the Markov chain.
        :param iterations: Number of iterations.
        :param device: CPU or GPU device.
        :return: List of sampled states.
        '''
        current_state = initial_state
        samples = []

        if device == 'cpu':
            for _ in range(iterations):
                proposed_state = create_proposal_dist(current_state)
                acceptance_ratio = create_target_dist(proposed_state) / create_target_dist(current_state)

                if np.random.rand() < acceptance_ratio:
                    current_state = proposed_state

                samples.append(current_state)
        else:
            for _ in range(iterations):
                proposed_state = create_proposal_dist(current_state)
                acceptance_ratio = create_target_dist(proposed_state) / create_target_dist(current_state)

                if cp.random.rand() < acceptance_ratio:
                    current_state = proposed_state

                samples.append(current_state)

        return samples

    @staticmethod
    def sequence_prob(o: np.ndarray | cp.ndarray,
                      a: np.ndarray | cp.ndarray,
                      b: np.ndarray | cp.ndarray,
                      pi: np.ndarray | cp.ndarray,
                      device: str="cpu") -> float:
        '''
        Calculates the probability of a sequence of observable symbols using the forward algorithm.
        :param o: Observable symbol sequence (1-dimensional ndarray).
        :param a: Transition probability matrix (2-dimensional ndarray).
        :param b: Emission probability matrix (2-dimensional ndarray).
        :param pi: Initial state distribution (1-dimensional ndarray).
        :param device: CPU or GPU device.
        :return: Probability of the observable sequence.
        '''
        n = a.shape[0]
        t = len(o)

        if device == 'cpu':
            no = o
            na = a
            nb = b
            npi = pi

            if isinstance(no, cp.ndarray):
                no = cp.asnumpy(no)

            if isinstance(na, cp.ndarray):
                na = cp.asnumpy(na)

            if isinstance(nb, cp.ndarray):
                nb = cp.asnumpy(nb)

            if isinstance(npi, cp.ndarray):
                npi = cp.asnumpy(npi)

            alpha = np.zeros((t, n))

            for i in range(n):
                alpha[0, i] = npi[i] * nb[i, no[0]]

            for i in range(1, t):
                for j in range(n):
                    alpha[i, j] = np.sum(alpha[i - 1] * na[:, j]) * nb[j, no[i]]

            return float(np.sum(alpha[t - 1, :]))
        else:
            co = o
            ca = a
            cb = b
            cpi = pi

            if isinstance(co, np.ndarray):
                co = cp.asarray(co)

            if isinstance(ca, np.ndarray):
                ca = cp.asarray(ca)

            if isinstance(cb, np.ndarray):
                cb = cp.asarray(cb)

            if isinstance(cpi, np.ndarray):
                cpi = cp.asarray(cpi)

            alpha = cp.zeros((t, n))

            for i in range(n):
                alpha[0, i] = cpi[i] * cb[i, co[0]]

            for i in range(1, t):
                for j in range(n):
                    alpha[i, j] = cp.sum(alpha[i - 1] * ca[:, j]) * cb[j, co[i]]

            return float(cp.sum(alpha[t - 1, :]))

    @staticmethod
    def stationary_distribution(t: np.ndarray | cp.ndarray,
                                device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Calculate the stationary distribution of a Markov chain.
        :param t: Transition probability matrix (2-dimensional ndarray).
        :param device: CPU or GPU device.
        :return: Stationary distribution.
        '''
        if device == 'cpu':
            nt = t

            if isinstance(nt, cp.ndarray):
                nt = cp.asnumpy(nt)

            n = nt.shape[0]

            a = np.append(nt.T - np.identity(n), np.ones((1, n)), axis=0)
            b = np.array([0] * n + [1])
            pi = np.linalg.solve(a, b)

            return pi
        else:
            ct = t

            if isinstance(ct, np.ndarray):
                ct = cp.asarray(ct)

            n = ct.shape[0]

            a = cp.append(ct.T - cp.identity(n), cp.ones((1, n)), axis=0)
            b = cp.array([0] * n + [1])
            pi = cp.linalg.solve(a, b)

            return pi

    @staticmethod
    def transition_prob_matrix(states: list,
                               transitions: list,
                               device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Create a transition probability matrix for a Markov chain.
        :param states: List of states.
        :param transitions: List of tuples indicating transitions (state from, state to).
        :param device: CPU or GPU device.
        :return: Transition probability matrix.
        '''
        n = len(states)
        p = np.zeros((n, n)) if device == 'cpu' else cp.zeros((n, n))
        state_indices = {state: index for index, state in enumerate(states)}

        for state_from, state_to in transitions:
            p[state_indices[state_from], state_indices[state_to]] += 1

        for i in range(n):
            p[i, :] /= p[i, :].sum()

        return p