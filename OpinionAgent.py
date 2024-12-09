import numpy as np
from random import Random
import mesa
import matplotlib.pyplot as plt

CONFIG = {
    "N": 200,              # Number of agents
    "tau": 0.2,             # Threshold for opinion difference to interact
    "mu": 0.2,              # Adjustment parameter for opinion change
    "steps": 2000,         # Number of simulation steps
    "seed": 42,           # Random seed for reproducibility
    "threshold": 0.5        # Proximity of opinions to count to the same cluster
}

class OpinionAgent(mesa.Agent):
    
    def __init__(self, unique_id, model, initial_opinion=None):
        """
        Create a new agent with an initial opinion
        
        Parameters:
        - unique_id: Unique identifier for the agent (is automatically assigned by Mensa)
        - model: Model the agent is part of
        - initial_opinion: Starting opinion (random if None)
        """
        super().__init__(model)
        # Initialize opinion randomly (with the seed) between -1 and 1 if not provided
        self.opinion = (
            initial_opinion
            if initial_opinion is not None
            else self.model.random.uniform(-1, 1)
        )

class OpinionDynamicsModel(mesa.Model):

    def __init__(self, N=CONFIG["N"], tau=CONFIG["tau"], mu=CONFIG["mu"], seed=CONFIG["seed"]):
        """
        Create a new model with a given number of agents
        
        Parameters:
        - N: Number of agents
        - tau: Threshold for opinion difference to trigger interaction (x > 0)
        - mu: Adjustment parameter for opinion change (0 < x <= 0.5)
        """
        super().__init__(seed=seed)
        
        # Model parameters
        self.num_agents = N
        self.tau = tau
        self.mu = mu
        self.random = Random(seed)
        
        # Step counter
        self.current_step = 0
        
        # Track opinion history for analysis and graphical display
        self.opinion_history = []

        # Create agents
        for i in range(self.num_agents):
            agent = OpinionAgent(unique_id=i, model=self)
            self.agents.add(agent) # Adds agents to the list
        
        # Datacollector for tracking model-level data
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "MeanOpinion": lambda m: np.mean([a.opinion for a in m.agents]),
                "OpinionStdDev": lambda m: np.std([a.opinion for a in m.agents]),
                "NumClusters": self.count_clusters
            }
        )
    
    def count_clusters(self, threshold=CONFIG["threshold"]):
        """
        Count the number of opinion clusters
        
        Parameters:
        - threshold: How close opinions need to be to be in the same cluster
        
        Returns:
        - Number of clusters
        """
        # Get all agent opinions
        opinions = [agent.opinion for agent in self.agents]
        # Track assigned agents
        assigned = set()
        clusters = 0
        
        for i, base_opinion in enumerate(opinions):
            # Skip if already in a cluster
            if i in assigned:
                continue
            # Start a new cluster
            clusters += 1
            # Find and mark agents close to this opinion
            for j, other_opinion in enumerate(opinions):
                if i != j and abs(base_opinion - other_opinion) < threshold:
                    assigned.add(j)
        return clusters
    
    def step(self):     # Advance the model by one step
        self.current_step += 1
        
        # Calculate percentage of comparisons completed
        percentage_done = (self.current_step / CONFIG["steps"]) * 100
        
        # Select two random agents
        agent1, agent2 = self.random.sample(list(self.agents), 2)
       
        # Print comparison message
        print(f"{percentage_done:.0f}% ({self.current_step}/{CONFIG['steps']}): comparing agent {agent1.unique_id} with agent {agent2.unique_id}")
        
        # Check if opinion proximity is within margins for interaction
        if abs(agent1.opinion - agent2.opinion) <= self.tau:
            agent1.opinion += self.mu * (agent2.opinion - agent1.opinion)
            agent2.opinion += self.mu * (agent1.opinion - agent2.opinion)
        
        # Collect data for this step
        self.datacollector.collect(self)
        # Track opinion history
        self.opinion_history.append([agent.opinion for agent in self.agents])
    
    
    def run_simulation(self, steps=CONFIG["steps"]):
        for _ in range(steps):
            self.step()

def plot_results(model):
    """
    Create visualizations of the simulation results
    
    Parameters:
    - model: Completed OpinionDynamicsModel instance
    """
    # Retrieve collected data
    model_data = model.datacollector.get_model_vars_dataframe()
    opinion_history = np.array(model.opinion_history)
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    
    # 1. Opinion Distribution Histogram
    ax1.hist(opinion_history[0], bins=30, alpha=0.5, label='Initial')
    ax1.hist(opinion_history[-1], bins=30, alpha=0.5, label='Final')
    ax1.set_title('Opinion Distribution')
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Number of Agents')
    ax1.legend()
    
    # 2. Individual Opinion Trajectories
    for i in range(min(100, model.num_agents)):
        ax2.plot(opinion_history[:, i], alpha=0.3)
    ax2.set_title('Individual Opinion Trajectories')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Opinion')
    
    # 3. Mean Opinion Over Time
    ax3.plot(model_data['MeanOpinion'])
    ax3.set_title('Mean Opinion Over Time')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Mean Opinion')
    
    # 4. Number of Clusters Over Time
    ax4.plot(model_data['NumClusters'])
    ax4.set_title('Number of Clusters Over Time')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Number of Clusters')
    
    plt.tight_layout()
    plt.show()

def main():
    print(f"Mesa version: {mesa.__version__}")
    
    # Create and run the model
    model = OpinionDynamicsModel()
    model.run_simulation()
    
    # Visualize results
    plot_results(model)
    
    # Print final statistics
    final_data = model.datacollector.get_model_vars_dataframe()
    print("\nFinal Simulation Statistics:")
    print(f"Final Mean Opinion: {final_data['MeanOpinion'].iloc[-1]:.4f}")
    print(f"Final Opinion Standard Deviation: {final_data['OpinionStdDev'].iloc[-1]:.4f}")
    print(f"Final Number of Clusters: {final_data['NumClusters'].iloc[-1]}")

if __name__ == "__main__":
    main()