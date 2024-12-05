import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class OpinionDynamicsSimulation:
    def __init__(self, N=1000, tau=0.5, mu=0.3):
        """
        Initialize the Opinion Dynamics Simulation
        
        Parameters:
        - N: Number of agents
        - tau: Threshold for opinion difference to trigger interaction
        - mu: Adjustment parameter for opinion change
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Initialize agent opinions randomly between 0 and 1
        self.N = N
        self.opinions = np.random.uniform(0, 1, N)
        
        # Simulation parameters
        self.tau = tau  # Threshold for interaction
        self.mu = mu    # Adjustment parameter
        
        # Track simulation statistics
        self.opinion_history = [self.opinions.copy()]
        self.cluster_history = []
    
    def find_clusters(self, threshold=0.1):
        """
        Identify clusters in the opinion space
        
        Parameters:
        - threshold: How close opinions need to be to be in the same cluster
        
        Returns:
        - List of clusters (groups of agent indices)
        """
        # Create a copy of opinions to avoid modifying original
        opinions = self.opinions.copy()
        clusters = []
        
        while len(opinions) > 0:
            # Start a new cluster with the first remaining opinion
            cluster = [0]
            base_opinion = self.opinions[0]
            
            # Remove the first opinion from the list to check
            opinions = opinions[1:]
            
            # Find all opinions close to the base opinion
            for i in range(len(opinions)):
                if abs(opinions[i] - base_opinion) < threshold:
                    cluster.append(i + 1)
                    # Remove the matched opinion
                    opinions = np.delete(opinions, i)
            
            clusters.append(cluster)
        
        return clusters
    
    def update(self):
        """
        Perform one time step of the opinion dynamics simulation
        """
        # Randomly pick two agents
        agent1_idx = np.random.randint(0, self.N)
        agent2_idx = np.random.randint(0, self.N)
        
        # Ensure agents are different
        while agent1_idx == agent2_idx:
            agent2_idx = np.random.randint(0, self.N)
        
        # Get their current opinions
        x1 = self.opinions[agent1_idx]
        x2 = self.opinions[agent2_idx]
        
        # Check if their opinions are close enough to interact
        if abs(x1 - x2) <= self.tau:
            # Adjust opinions
            self.opinions[agent1_idx] += self.mu * (x2 - x1)
            self.opinions[agent2_idx] += self.mu * (x1 - x2)
        
        # Track opinion history and clusters
        self.opinion_history.append(self.opinions.copy())
        self.cluster_history.append(len(self.find_clusters()))
    
    def simulate(self, timesteps=20000):
        """
        Run the full simulation
        
        Parameters:
        - timesteps: Number of time steps to simulate
        """
        for _ in range(timesteps):
            self.update()
    
    def plot_results(self):
        """
        Create visualizations of the simulation results
        """
        # Create a figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. Opinion Distribution Histogram at Start and End
        ax1.hist(self.opinion_history[0], bins=30, alpha=0.5, label='Initial')
        ax1.hist(self.opinion_history[-1], bins=30, alpha=0.5, label='Final')
        ax1.set_title('Opinion Distribution')
        ax1.set_xlabel('Opinion')
        ax1.set_ylabel('Number of Agents')
        ax1.legend()
        
        # 2. Opinion Trajectories
        opinion_array = np.array(self.opinion_history)
        for i in range(min(100, self.N)):  # Plot trajectories for first 100 agents
            ax2.plot(opinion_array[:, i], alpha=0.3)
        ax2.set_title('Individual Opinion Trajectories')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Opinion')
        
        # 3. Number of Clusters Over Time
        ax3.plot(self.cluster_history)
        ax3.set_title('Number of Clusters Over Time')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Number of Clusters')
        
        plt.tight_layout()
        plt.show()

# Main execution
def main():
    # Create and run the simulation
    sim = OpinionDynamicsSimulation(N=1000, tau=0.5, mu=0.3)
    sim.simulate(timesteps=20000)
    
    # Plot results
    sim.plot_results()
    
    # Additional statistics
    final_opinions = sim.opinions
    print(f"Final Opinion Statistics:")
    print(f"Mean: {np.mean(final_opinions):.4f}")
    print(f"Standard Deviation: {np.std(final_opinions):.4f}")
    print(f"Number of Final Clusters: {len(sim.find_clusters())}")

if __name__ == "__main__":
    main()
