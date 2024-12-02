import numpy as np
import matplotlib.pyplot as plt

class OpinionDynamicsModel:
    def __init__(self, grid_size=50, initial_opinion_spread=0.2, 
                 influence_radius=2, convergence_threshold=0.01, 
                 seed=None, time_dependent_seed=False):
        """
        Initialize the opinion dynamics model with seed options
        
        Parameters:
        - seed: Fixed seed for reproducibility or None
        - time_dependent_seed: If True, uses current time for randomness
        """
        # Set up random number generator
        if time_dependent_seed:
            # Use current time for seed
            seed = int(np.random.default_rng().random() * 1000000)
        
        # Set the seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize opinions with some random variation
        self.grid_size = grid_size
        self.opinions = np.random.normal(
            loc=0.5,  # mean opinion
            scale=initial_opinion_spread, 
            size=(grid_size, grid_size)
        )
        
        # Clip opinions to [0, 1] range
        self.opinions = np.clip(self.opinions, 0, 1)
        
        self.influence_radius = influence_radius
        self.convergence_threshold = convergence_threshold
        
    def update(self, model_type='average'):
        """
        Update opinions based on neighborhood
        
        model_types:
        - 'average': Simple averaging of opinions
        - 'bounded_confidence': Agents only influenced by similar opinions
        """
        new_opinions = self.opinions.copy()
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Define neighborhood
                x_min = max(0, x - self.influence_radius)
                x_max = min(self.grid_size, x + self.influence_radius + 1)
                y_min = max(0, y - self.influence_radius)
                y_max = min(self.grid_size, y + self.influence_radius + 1)
                
                # Extract neighborhood
                neighborhood = self.opinions[x_min:x_max, y_min:y_max]
                
                if model_type == 'average':
                    # Simple average of neighborhood
                    new_opinions[x, y] = np.mean(neighborhood)
                
                elif model_type == 'bounded_confidence':
                    # Only average with opinions within a confidence bound
                    confidence_bound = 0.2
                    similar_agents = np.abs(neighborhood - self.opinions[x, y]) <= confidence_bound
                    if np.any(similar_agents):
                        new_opinions[x, y] = np.mean(neighborhood[similar_agents])
        
        # Calculate total change
        change = np.abs(new_opinions - self.opinions).mean()
        
        # Update opinions
        self.opinions = new_opinions
        
        return change
    
    def run_simulation(self, max_iterations=100, model_type='average'):
        """
        Run the opinion dynamics simulation
        
        Returns:
        - Final opinion grid
        - Opinion evolution over time
        """
        opinion_history = [self.opinions.copy()]
        
        for _ in range(max_iterations):
            change = self.update(model_type)
            opinion_history.append(self.opinions.copy())
            
            # Stop if change is below threshold
            if change < self.convergence_threshold:
                break
        
        return self.opinions, opinion_history
    
    def visualize(self, opinion_history):
        """
        Visualize opinion evolution
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()
        
        # Select a few snapshots of opinion evolution
        snapshot_indices = [0, len(opinion_history)//4, 
                            len(opinion_history)//2, 
                            3*len(opinion_history)//4, 
                            -1]
        
        for i, idx in enumerate(snapshot_indices):
            im = axs[i].imshow(opinion_history[idx], 
                               cmap='coolwarm', 
                               vmin=0, vmax=1)
            axs[i].set_title(f'Iteration {idx}')
            plt.colorbar(im, ax=axs[i])
        
        plt.tight_layout()
        plt.show()

# Example usage demonstrations
def demonstrate_seeding():
    print("Demonstration of Seeding Options:")
    
    # Fixed seed for reproducibility
    print("\n1. Fixed Seed Simulation:")
    model_fixed = OpinionDynamicsModel(seed=42)
    final_fixed, history_fixed = model_fixed.run_simulation()
    model_fixed.visualize(history_fixed)
    
    # Time-dependent seed
    print("\n2. Time-Dependent Seed Simulation:")
    model_time = OpinionDynamicsModel(time_dependent_seed=True)
    final_time, history_time = model_time.run_simulation()
    model_time.visualize(history_time)
    
    # No seed (completely random)
    print("\n3. No Seed (Completely Random) Simulation:")
    model_random = OpinionDynamicsModel()
    final_random, history_random = model_random.run_simulation()
    model_random.visualize(history_random)

# Run the demonstration
demonstrate_seeding()
