# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:29:43 2019

@author: jkpvsz
"""
import numpy as np
from itertools import combinations

__operators__ = ['random_search', 'random_sample', 'rayleigh_flight',
                 'inertial_pso', 'constricted_pso', 'levy_flight', 'mutation_de',
                 'spiral_dynamic', 'central_force', 'gravitational_search']
__crossover__ = ['binomial_crossover_de', 'exponential_crossover_de']
__selection__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']

#     Operator call name, dictionary with default parameters, default selector
__simple_heuristics__ = [
    ("local_random_walk", dict(probability=0.75, scale=1.0), "greedy"),                       # 1
    ('random_search', dict(scale=0.01), "greedy"),                                               # 2
    ("random_sample", dict(), "greedy"),                                                            # 3
    ("rayleigh_flight", dict(scale=0.01), "greedy"),                                             # 4
    ("levy_flight", dict(scale=1.0, beta=1.5), "greedy"),                                     # 5
    ("mutation_de", dict(scheme=("current-to-best", 1), factor=1.0), "greedy"),               # 6
    ('binomial_crossover_de', dict(crossover_rate=0.5), "greedy"),                               # 7
    ("exponential_crossover_de", dict(crossover_rate=0.5), "greedy"),                            # 8
    ("firefly", dict(epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0), "greedy"),     # 9
    ("inertial_pso", dict(inertial=0.7, self_conf=1.54, swarm_conf=1.56), "all"),          # 10
    ("constriction_pso", dict(kappa=1.0, self_conf=2.54, swarm_conf=2.56), "all"),          # 11
    ("gravitational_search", dict(alpha=0.02, epsilon=1e-23), "all"),                         # 12
    ("central_force", dict(gravity=0.001, alpha=0.001, beta=1.5, dt=1.0), "all"),       # 13
    ("spiral_dynamic", dict(radius=0.9, angle=22.5, sigma=0.1), "all")                      # 14
]


# %% --------------------------------------------------------------------------
class Population():
    # Internal variables
    iteration = 1
    rotation_matrix = []

    # Parameters per selection method    
    metropolis_temperature = 1000.0
    metropolis_rate = 0.01
    metropolis_boltzmann = 1.0
    probability_selection = 0.5

    # Class initialisation
    # -------------------------------------------------------------------------
    def __init__(self, problem_function, boundaries, num_agents=30,
                 is_constrained=True):
        # Read problem, it must be a callable function
        self.problem_function = problem_function

        # boundaries must be a tuple of np.ndarrays

        # Read number of variables or dimension 
        self.num_dimensions = len(boundaries[0])

        # Read the upper and lower boundaries of search space
        self.lower_boundaries = boundaries[0]
        self.upper_boundaries = boundaries[1]
        self.span_boundaries = self.upper_boundaries - self.lower_boundaries
        self.centre_boundaries = (self.upper_boundaries + self.lower_boundaries) / 2

        # Read number of agents in population
        self.num_agents = num_agents

        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        self.fitness = np.full(self.num_agents, np.nan)

        # General fitness measurements        
        self.global_best_position = np.full(self.num_dimensions, np.nan)
        self.global_best_fitness = float('inf')

        self.current_best_position = np.full(self.num_dimensions, np.nan)
        self.current_best_fitness = float('inf')
        self.current_worst_position = np.full(self.num_dimensions, np.nan)
        self.current_worst_fitness = -float('inf')

        self.particular_best_positions = np.full(
            (self.num_agents, self.num_dimensions), np.nan)
        self.particular_best_fitness = np.full(self.num_agents, np.nan)

        self.previous_positions = np.full((self.num_agents,
                                           self.num_dimensions), np.nan)
        self.previous_velocities = np.full((self.num_agents,
                                            self.num_dimensions), np.nan)
        self.previous_fitness = np.full(self.num_agents, np.nan)

        self.is_constrained = is_constrained

        # TODO Add capability for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions

    # %% ----------------------------------------------------------------------
    #    BASIC TOOLS
    # -------------------------------------------------------------------------

    # Generate a string containing the current state of the population
    # -------------------------------------------------------------------------
    def get_state(self):
        return ("x_best = " + str(self.__rescale_back(
            self.global_best_position)) + ", with f_best = " +
                str(self.global_best_fitness))

    # Rescale all agents from [-1,1] to [lower,upper]
    # -------------------------------------------------------------------------
    def get_population(self):
        rescaled_positions = np.tile(self.centre_boundaries,
                                     (self.num_agents, 1)) + self.positions * np.tile(
            self.span_boundaries / 2, (self.num_agents, 1))

        return rescaled_positions

    # Rescale all agents from [lower,upper] to [-1,1]
    # -------------------------------------------------------------------------
    def set_population(self, positions):
        rescaled_positions = 2 * (positions - np.tile(
            self.centre_boundaries, (self.num_agents, 1))) / np.tile(
            self.span_boundaries, (self.num_agents, 1))

        return rescaled_positions

    # Evaluate population positions in the problem function
    # -------------------------------------------------------------------------
    def evaluate_fitness(self):
        for agent in range(self.num_agents):
            self.fitness[agent] = self.problem_function(
                self.__rescale_back(self.positions[agent, :]))

    # Update population positions according to a selection scheme
    # -------------------------------------------------------------------------
    def update_population(self, selection_method="all"):
        # Update population positons, velocities, and fitness        
        for agent in range(self.num_agents):
            if getattr(self, "_" + selection_method + "_selection")(
                    self.fitness[agent], self.previous_fitness[agent]):
                # if new positions are improved, then update past register ...
                self.previous_fitness[agent] = self.fitness[agent]
                self.previous_positions[agent, :] = self.positions[agent, :]
                self.previous_velocities[agent, :] = self.velocities[agent, :]
            else:
                # ... otherwise,return to previous values
                self.fitness[agent] = self.previous_fitness[agent]
                self.positions[agent, :] = self.previous_positions[agent, :]
                self.velocities[agent, :] = self.previous_velocities[agent, :]

        # Update the current best and worst positions (forced to greedy)
        self.current_best_position = self.positions[self.fitness.argmin(), :]
        self.current_best_fitness = self.fitness.min()

        self.current_worst_position = self.positions[self.fitness.argmax(), :]
        self.current_worst_fitness = self.fitness.max()

    # Update particular positions according to a selection scheme
    # -------------------------------------------------------------------------
    def update_particular(self, selection_method="greedy"):
        for agent in range(self.num_agents):
            if getattr(self, "_" + selection_method + "_selection")(
                    self.fitness[agent], self.particular_best_fitness[agent]):
                self.particular_best_fitness[agent] = self.fitness[agent]
                self.particular_best_positions[agent, :] = self.positions[agent, :]

    # Update global position according to a selection scheme
    # -------------------------------------------------------------------------
    def update_global(self, selection_method="greedy"):
        # Perform particular updating
        self.update_particular(selection_method)

        # Read current global best agent
        candidate_position = self.particular_best_positions[ \
                             self.particular_best_fitness.argmin(), :]
        candidate_fitness = self.particular_best_fitness.min()
        if getattr(self, "_" + selection_method + "_selection")(
                candidate_fitness, self.global_best_fitness) or np.isinf(
            candidate_fitness):
            self.global_best_position = candidate_position
            self.global_best_fitness = candidate_fitness

            # %% ----------------------------------------------------------------------

    #    INITIALISATORS
    # -------------------------------------------------------------------------

    # Initialise population using a random uniform in [-1,1]
    # -------------------------------------------------------------------------
    def initialise_uniformly(self):
        for agent in range(0, self.num_agents):
            self.positions[agent] = np.random.uniform(-1, 1, self.num_dimensions)

            # TODO Add more initialisation operators like grid, boundary, etc.

    # %% ----------------------------------------------------------------------
    #    PERTURBATORS
    # -------------------------------------------------------------------------

    # Random sample  
    # -------------------------------------------------------------------------
    def random_sample(self):
        # Create random positions using random numbers between -1 and 1
        self.positions = np.random.uniform(-1, 1, (self.num_agents, self.num_dimensions))

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Random Walk   
    # -------------------------------------------------------------------------
    def random_search(self, scale=0.01):
        # Move each agent using uniform random displacements
        self.positions += scale * np.random.uniform(-1, 1, (self.num_agents, self.num_dimensions))

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

        # Rayleigh Flight

    # -------------------------------------------------------------------------
    def rayleigh_flight(self, scale=0.01):
        # Move each agent using gaussian random displacements
        self.positions += scale * np.random.standard_normal((self.num_agents, self.num_dimensions))

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Lévy flight (Mantegna's algorithm)
    # -------------------------------------------------------------------------
    def levy_flight(self, scale=1.0, beta=1.5):
        # Calculate x's std dev (Mantegna's algorithm)
        sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (np.math.gamma((1 + beta) / 2) * beta *
                                                                         (2 ** ((beta - 1) / 2)))) ** (1 / beta)

        # Determine x and y using normal distributions with sigma_y = 1
        x = sigma * np.random.standard_normal((self.num_agents, self.num_dimensions))
        y = np.abs(np.random.standard_normal((self.num_agents, self.num_dimensions)))

        # Calculate the random number with levy stable distribution
        levy_random = x / (y ** (1 / beta))

        # Determine z as an additional normal random number
        z = np.random.standard_normal((self.num_agents, self.num_dimensions))

        # Move each agent using levy random displacements
        self.positions += scale * z * levy_random * (self.positions - np.tile(self.global_best_position,
                                                                              (self.num_agents, 1)))

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Inertial PSO movement
    # -------------------------------------------------------------------------
    def inertial_pso(self, inertial=0.7, self_conf=1.54, swarm_conf=1.56):
        # Determine random numbers
        r_1 = self_conf * np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = swarm_conf * np.random.rand(self.num_agents, self.num_dimensions)

        # Find new velocities 
        self.velocities = inertial * self.velocities + r_1 * (
                self.particular_best_positions - self.positions) + r_2 * (
                                  np.tile(self.global_best_position, (self.num_agents, 1)) - self.positions)

        # Move each agent using velocity's information
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Constricted PSO movement  
    # -------------------------------------------------------------------------              
    def constriction_pso(self, kappa=1.0, self_conf=2.54, swarm_conf=2.56):
        # Find the constriction factor chi using phi
        phi = self_conf + swarm_conf
        if phi > 4: chi = 2 * kappa / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        else: chi = np.sqrt(kappa)

        # Determine random numbers
        r_1 = self_conf * np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = swarm_conf * np.random.rand(self.num_agents, self.num_dimensions)

        # Find new velocities 
        self.velocities = chi * (self.velocities + r_1 * (self.particular_best_positions - self.positions) +
                                 r_2 * (np.tile(self.global_best_position, (self.num_agents, 1)) - self.positions))

        # Move each agent using velocity's information
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Differential evolution mutation
    # -------------------------------------------------------------------------
    def mutation_de(self, scheme=("current-to-best", 1), factor=1.0):
        # Read the kind of expression and the number of sums to use
        expression, num_rands = scheme

        # Create mutants using the expression provided in scheme
        if expression == "rand":
            mutant = self.positions[np.random.permutation(self.num_agents), :]

        elif expression == "best":
            mutant = np.tile(self.global_best_position, (self.num_agents, 1))

        elif expression == "current":
            mutant = self.positions

        elif expression == "current-to-best":
            mutant = self.positions + factor * (np.tile(self.global_best_position, (self.num_agents, 1)) -
                                                self.positions[np.random.permutation(self.num_agents), :])

        elif expression == "rand-to-best":
            mutant = self.positions[np.random.permutation(self.num_agents), :] + factor * (np.tile(
                self.global_best_position,(self.num_agents, 1)) - self.positions[np.random.permutation(
                self.num_agents), :])

        elif expression == "rand-to-bestandcurrent":
            mutant = self.positions[np.random.permutation(self.num_agents), :] + factor * \
                     (np.tile(self.global_best_position, (self.num_agents, 1)) - self.positions[np.random.permutation(
                         self.num_agents), :] +self.positions[np.random.permutation(self.num_agents), :]-self.positions)
        else:
            mutant = []
            print('[Error] Check de_mutation_scheme!')

        # Add random parts according to num_rands
        if num_rands >= 0:
            for _ in range(num_rands):
                mutant += factor * (self.positions[np.random.permutation(self.num_agents), :] -
                               self.positions[np.random.permutation(self.num_agents), :])
        else:
            print('[Error] Check de_mutation_scheme!')

        # Replace mutant population in the current one
        self.positions = mutant

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Binomial crossover from Differential evolution
    # -------------------------------------------------------------------------
    def binomial_crossover_de(self, crossover_rate=0.5):
        # Define indices
        indices = np.tile(np.arange(self.num_dimensions), (self.num_agents, 1))

        # Permute indices per dimension
        rand_indices = np.vectorize(np.random.permutation, signature='(n)->(n)')(indices)

        # Calculate the NOT condition (because positions were already updated!)
        condition = np.logical_not((indices == rand_indices) | (np.random.rand(self.num_agents, self.num_dimensions) <=
                                                                crossover_rate))

        # Reverse the ones to their previous positions 
        self.positions[condition] = self.previous_positions[condition]

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Exponential crossover from Differential evolution
    # -------------------------------------------------------------------------
    def exponential_crossover_de(self, crossover_rate=0.5):
        # Perform the exponential crossover procedure
        for agent in range(self.num_agents):
            for dim in range(self.num_dimensions):
                # Initialise L and choose a random index n
                exp_var = 0
                n = np.random.randint(self.num_dimensions)
                while True:
                    # Increase L and check the exponential crossover condition
                    exp_var += 1
                    if np.logical_not((np.random.rand() < crossover_rate) and (exp_var < self.num_dimensions)): break

                # Perform the crossover if the following condition is met
                if dim not in [(n + x) % self.num_dimensions for x in range(exp_var)]:
                    self.positions[agent, dim] = self.previous_positions[agent, dim]

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Local random walk from Cuckoo Search
    # -------------------------------------------------------------------------
    def local_random_walk(self, probability=0.75, scale=1.0):
        # Determine random numbers
        r_1 = np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = np.random.rand(self.num_agents, self.num_dimensions)

        # Move positions with a displacement due permutations and probabilities
        self.positions += scale * r_1 * (self.positions[np.random.permutation(self.num_agents), :] -
                                         self.positions[np.random.permutation(self.num_agents), :]) * \
                          np.heaviside(r_2 - probability, 0)

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Deterministic/Stochastic Spiral dynamic
    # -------------------------------------------------------------------------
    def spiral_dynamic(self, radius=0.9, angle=22.5, sigma=0.1):
        # Update rotation matrix 
        self.__get_rotation_matrix(np.deg2rad(angle))

        for agent in range(self.num_agents):
            random_radii = np.random.uniform(radius - sigma, radius + sigma, self.num_dimensions)
            self.positions[agent, :] = self.global_best_position + random_radii * \
                                np.matmul(self.rotation_matrix, (self.positions[agent, :] - self.global_best_position))

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Firefly (generalised)
    # -------------------------------------------------------------------------
    def firefly(self, epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0):
        # Determine epsilon values
        if epsilon == "gaussian":
            epsilon_value = np.random.standard_normal((self.num_agents, self.num_dimensions))
        elif epsilon == "uniform":
            epsilon_value = np.random.uniform(-0.5, 0.5, (self.num_agents, self.num_dimensions))
        else:
            epsilon_value = []
            print("epsilon value is not valid: 'uniform' | 'gaussian' ")

        # Initialise delta or difference between two positions
        difference_positions = np.zeros((self.num_agents, self.num_dimensions))

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all vectorial distances with respect to agent
            delta = self.positions[indices, :] - np.tile(self.positions[agent, :], (self.num_agents - 1, 1))

            # Determine differences between lights
            delta_lights = np.tile((self.fitness[indices] - np.tile(self.fitness[agent],
                                    (1, self.num_agents - 1))).transpose(), (1, self.num_dimensions))

            # Find the total attraction for each agent
            difference_positions[agent, :] = np.sum(np.heaviside(-delta_lights, 0) * delta *
                                                    np.exp(-gamma * np.linalg.norm(delta, 2, 1) ** 2), 0)

        # Move fireflies according to their attractions
        self.positions += alpha * epsilon_value + beta * difference_positions

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Central Force Optimisation (CFO)
    # -------------------------------------------------------------------------
    def central_force(self, gravity=0.001, alpha=0.001, beta=1.5, dt=1.0):
        # Initialise acceleration
        acceleration = np.zeros((self.num_agents, self.num_dimensions))

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all masses differences with respect to agent
            delta_masses = self.fitness[indices] - np.tile(self.fitness[agent], (1, self.num_agents - 1))

            # Determine all vectorial distances with respect to agent
            delta_positions = self.positions[indices, :] - np.tile(self.positions[agent, :], (self.num_agents - 1, 1))

            distances = np.linalg.norm(delta_positions, 2, 1)

            # Find the quotient part    ! -> - delta_masses (cz minimisation)
            quotient = np.heaviside(-delta_masses, 0) * (np.abs(delta_masses) ** alpha) / (distances ** beta)

            # Determine the acceleration for each agent
            acceleration[agent, :] = gravity * np.sum(delta_positions * np.tile(quotient.transpose(),
                                                                                (1, self.num_dimensions)), 0)

        self.positions += 0.5 * acceleration * (dt ** 2)

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Gravitational Search Algorithm (GSA) simplified
    # -------------------------------------------------------------------------
    def gravitational_search(self, gravity=1, alpha=0.02, epsilon=1e-23):
        # Initialise acceleration
        acceleration = np.zeros((self.num_agents, self.num_dimensions))

        # Determine the gravitational constant
        gravitation = gravity * np.exp(- alpha * self.iteration)

        # Determine mass for each agent
        raw_masses = (self.fitness - np.tile(self.current_worst_fitness, (1, self.num_agents)))
        masses = (raw_masses / np.sum(raw_masses)).reshape(self.num_agents)

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all vectorial distances with respect to agent
            delta_positions = self.positions[indices, :] - np.tile(self.positions[agent, :], (self.num_agents - 1, 1))

            quotient = masses[indices] / (np.linalg.norm(delta_positions, 2, 1) + epsilon)

            # Force interaction
            force_interaction = gravitation * np.tile(quotient.reshape(self.num_agents - 1, 1),
                                                      (1, self.num_dimensions)) * delta_positions

            # Acceleration
            acceleration[agent, :] = np.sum(np.random.rand(self.num_agents - 1, self.num_dimensions) *
                                            force_interaction, 0)

        # Update velocities
        self.velocities = np.random.rand(self.num_agents, self.num_dimensions) * self.velocities + acceleration

        # Update positions
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained: self.__check_simple_constraints()

    # Genetic Algorithm (GA): Crossover
    # -------------------------------------------------------------------------
    def ge_crossover(self, mating_pool_factor=0.1, mating_selection="natural", elitism_factor=0.01):
        # Mating pool size
        num_mates = np.round(mating_pool_factor * self.num_agents)

        # Number of elite agents
        num_elite = np.ceil(elitism_factor * self.num_agents)

        # Number of offsprings
        num_offsprings = self.num_agents - num_mates - num_elite

        # Get the mating pool using the selection strategy specified
        mating_pool_indices = self._ga_natural_selection(num_mates)

        # Identify offspring indices
        offspring_indices = np.setdiff1d(np.arange(self.num_agents), mating_pool_factor, True)

        # Perform crossover
        #for offspring in range(num_offsprings):
            # Choose two parents from mating pool using a selection strategy

    # Genetic Algorithm (GA): Selection strategies
    # -------------------------------------------------------------------------
    # -> Natural selection to obtain the mating pool
    def _ga_natural_selection(self, num_mates):
        # Sort population according to its fitness values
        sorted_indices = np.argsort(self.fitness)

        # Return indices corresponding mating pool
        return sorted_indices[:num_mates]

    # -> Tournament pairing
    def _ga_tournament_pairing(self, mating_pool, tournament_size=2, probability=1.0):
        # Calculate probabilities
        probabilities = probability * (1 - probability) ** np.arange(tournament_size)

        # Initialise the mother and father indices
        mate_indices = np.full(2, np.nan)

        # Perform tournaments until all mates are selected
        mate = 0
        while mate < 2:
            # Choose tournament candidates
            random_indices = np.random.permutation(self.num_agents)[:tournament_size]

            # Determine the candidate fitness values
            candidates_indices = random_indices[np.argsort(self.fitness[random_indices])]

            # Find the best according to its fitness and probability
            possible_winner = candidates_indices[np.random.rand(tournament_size) < probabilities]
            if possible_winner.size > 0:
                mate_indices[mate] = possible_winner[0]
                mate += 1

        # Return the mating pool and its fitness
        return mate_indices

    # -> Roulette Wheel (Rank Weighting) Selection
    def _ga_rank_weighting_pairing(self, mating_pool):
        # Determine the probabilities
        probabilities = (mating_pool.size - np.arange(mating_pool.size)) / np.sum(np.arange(mating_pool.size) + 1)

        # Perform the roulette wheel selection and return parents
        return np.searchsorted(np.cumsum(probabilities), np.random.rand(2))

    # %% ----------------------------------------------------------------------
    #    INTERNAL METHODS (avoid using them outside)
    # -------------------------------------------------------------------------

    # Check simple constraints if self.is_constrained = True
    # -------------------------------------------------------------------------
    def __check_simple_constraints(self):
        # Check if agents are beyond lower boundaries
        low_check = self.positions < -1.
        if low_check.any():
            # Fix them
            self.positions[low_check] = -1.
            self.velocities[low_check] = 0.

        # Check if agents are beyond upper boundaries
        upp_check = self.positions > 1.
        if upp_check.any():
            # Fix them
            self.positions[upp_check] = 1.
            self.velocities[upp_check] = 0.

    # Rescale an agent from [-1,1] to [lower,upper] per dimension
    # -------------------------------------------------------------------------
    def __rescale_back(self, position):
        return self.centre_boundaries + position * (self.span_boundaries / 2)

    # Generate a N-D rotation matrix for a given angle
    # -------------------------------------------------------------------------
    def __get_rotation_matrix(self, angle=0.39269908169872414):
        # Initialise the rotation matrix
        rotation_matrix = np.eye(self.num_dimensions)

        # Find the combinations without repetions
        planes = list(combinations(range(self.num_dimensions), 2))

        # Create the rotation matrix
        for xy in range(len(planes)):
            # Read dimensions
            x, y = planes[xy]

            # (Re)-initialise a rotation matrix for each plane
            rotation_plane = np.eye(self.num_dimensions)

            # Assign corresponding values
            rotation_plane[x, y] = np.cos(angle)
            rotation_plane[y, y] = np.cos(angle)
            rotation_plane[x, y] = -np.sin(angle)
            rotation_plane[y, x] = np.sin(angle)

            rotation_matrix = np.matmul(rotation_matrix, rotation_plane)

        self.rotation_matrix = rotation_matrix

    # Greedy selection : new is better than old one
    # -------------------------------------------------------------------------
    def _greedy_selection(self, new, old):
        return new <= old

    # Metropolis selection : apply greedy selection and worst with a prob
    # -------------------------------------------------------------------------
    def _metropolis_selection(self, new, old):
        # It depends of metropolis_temperature, metropolis_rate, and iteration
        if new <= old:
            selection_condition = True
        else:
            if np.math.exp(-(new - old) / (self.metropolis_boltzmann * self.metropolis_temperature *
                                           ((1 - self.metropolis_rate) ** self.iteration) + 1e-23)) > np.random.rand():
                selection_condition = True
            else:
                selection_condition = False
        return selection_condition

        # Probabilistic selection : apply greedy and worst is chosen if rand < prob

    # -------------------------------------------------------------------------
    def _probabilistic_selection(self, new, old):
        # It depends of metropolis_temperature, metropolis_rate, and iteration
        if new <= old:
            selection_condition = True
        else:
            if np.random.rand() < self.probability_selection:
                selection_condition = True
            else:
                selection_condition = False
        return selection_condition

        # All selection : only new does matter

    # -------------------------------------------------------------------------
    def _all_selection(self, *args):
        return True

    # None selection : new does not matter
    # -------------------------------------------------------------------------
    def _none_selection(self, *args):
        return False