import numpy as np
import mesa
import matplotlib.pyplot as plt

class OpinionAgent(mesa.Agent):
    """
    An agent with a specific opinion that can interact with other agents
    """
    def __init__(self, unique_id, model, initial_opinion=None):
        """
        Create a new agent with an initial opinion
        
        Parameters:
        - unique_id: Unique identifier for the agent
        - model: Model the agent is part of
        - initial_opinion: Starting opinion (random if None)
        """
        super().__init__(unique_id, model)
        
        # Initialize opinion randomly between 0 and 1 if not provided
        self.opinion = initial_opinion if initial_opinion is not None else np.random.uniform(0, 1)
    
    def step(self):
        """
        Agent's step method - attempt to interact with another agent
        """
        # Randomly select another agent in the model
        other_agent = self.random.choice(self.model.schedule.agents)
        
        # Check if agents can interact based on opinion proximity
        if other_agent != self and abs(self.opinion - other_agent.opinion) <= self.model.tau:
            # Adjust opinions towards each other
            self.opinion += self.model.mu * (other_agent.opinion - self.opinion)
            other_agent.opinion += self.model.mu * (self.opinion - other_agent.opinion)

class OpinionDynamicsModel(mesa.Model):
    """
    Model of opinion dynamics with multiple agents
    """
    def __init__(self, N=1000, tau=0.5, mu=0.3):
        """
        Create a new model with a given number of agents
        
        Parameters:
        - N: Number of agents
        - tau: Threshold for opinion difference to trigger interaction
        - mu: Adjustment parameter for opinion change
        """
        # Create a scheduler to manage agent actions
        self.schedule = mesa.time.RandomActivation(self)
        
        # Model parameters
        self.num_agents = N
        self.tau = tau
        self.mu = mu
        
        # Random seed for reproducibility
        self.random.seed(42)
        
        # Track opinion history for analysis
        self.opinion_history = []
        
        # Create agents and add to schedule
        for i in range(self.num_agents):
            agent = OpinionAgent(i, self)
            self.schedule.add(agent)
        
        # Datacollector for tracking model-level data
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "MeanOpinion": lambda m: np.mean([a.opinion for a in m.schedule.agents]),
                "OpinionStdDev": lambda m: np.std([a.opinion for a in m.schedule.agents]),
                "NumClusters": self.count_clusters
            }
        )
    
    def count_clusters(self, threshold=0.1):
        """
        Count the number of opinion clusters
        
        Parameters:
        - threshold: How close opinions need to be to be in the same cluster
        
        Returns:
        - Number of clusters
        """
        # Get all agent opinions
        opinions = [agent.opinion for agent in self.schedule.agents]
        
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
    
    def step(self):
        """
        Advance the model by one step
        """
        # Take a step in the schedule (activates all agents)
        self.schedule.step()
        
        # Collect data for this step
        self.datacollector.collect(self)
        
        # Track opinion history
        self.opinion_history.append([agent.opinion for agent in self.schedule.agents])
    
    def run_simulation(self, steps=20000):
        """
        Run the full simulation for a given number of steps
        
        Parameters:
        - steps: Number of simulation steps to run
        """
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
    """
    Run the opinion dynamics simulation and visualize results
    """
    # Create and run the model
    model = OpinionDynamicsModel(N=1000, tau=0.5, mu=0.3)
    model.run_simulation(steps=20000)
    
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