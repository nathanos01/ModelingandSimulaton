import os
import numpy as np
from random import Random
import mesa
import matplotlib.pyplot as plt
import itertools
import pandas as pd

CONFIG = {
    "N": 100,               # Number of agents
    "steps": 2000,          # Number of simulation steps
    "seed": 1482,        # Seed for reproducibility
    "threshold": 0.3,        # Proximity of opinions to count to the same cluster

    # Calculation methode dependent parameters
    "mode": "single",        # "sweep" for parameter sweep, "single" for a single simulation
    # If mode = single
    "tau": 0.3,             # Threshold for opinion difference to interact [x>0]
    "mu": 0.5,              # Adjustment parameter for opinion change [0<x<=0.5]
    # If mode = sweep
    "tau_values": [0.1, 0.2, 0.3, 0.4, 0.5],  # Tau values for sweep
    "mu_values": [0.1, 0.2, 0.3, 0.4, 0.5]    # Mu values for sweep
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
        self.current_step = 0    # Step counter
        
        # Track opinion history for analysis and graphical display
        self.opinion_history = []

        # Create agents
        for i in range(self.num_agents):
            agent = OpinionAgent(unique_id=i, model=self)
            self.agents.add(agent)       # Adds agents to the list
        
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
        opinions = [agent.opinion for agent in self.agents]     # Get all agent opinions
        assigned = set()  # To track opinions already assigned to a cluster
        clusters = 0      # Cluster count

        for i, base_opinion in enumerate(opinions):
            if i in assigned:
                continue
            clusters += 1
            for j, other_opinion in enumerate(opinions):
                if i != j and abs(base_opinion - other_opinion) < threshold:
                    assigned.add(j)
        return clusters

    def step(self):
        """
        Advance the model by one step.
        """
        self.current_step += 1
        # Select two random agents to interact
        agent1, agent2 = self.random.sample(list(self.agents), 2)

        # Calculate percentage of comparisons completed and display comparison message
        #percentage_done = (self.current_step / CONFIG["steps"]) * 100
        #print(f"{percentage_done:.0f}% ({self.current_step}/{CONFIG['steps']}): comparing agent {agent1.unique_id} with agent {agent2.unique_id}")

        if abs(agent1.opinion - agent2.opinion) <= self.tau:
            # Update opinions based on the interaction
            agent1.opinion += self.mu * (agent2.opinion - agent1.opinion)
            agent2.opinion += self.mu * (agent1.opinion - agent2.opinion)

        # Collect data for this step and track opinion history
        self.datacollector.collect(self)
        self.opinion_history.append([agent.opinion for agent in self.agents])
    
    def run_simulation(self, steps=CONFIG["steps"]):
        for _ in range(steps):
            self.step()

def plot_results(model):
    """
    Create visualizations of the simulation results
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

def parameter_sweep(tau_values, mu_values, steps=CONFIG["steps"], N=CONFIG["N"], seed=CONFIG["seed"]):
    """
    Perform a parameter sweep over tau and mu values to analyze their impact on clustering.
    Returns: A DataFrame containing the results of the parameter sweep
    """
    results = []

    for tau, mu in itertools.product(tau_values, mu_values):
        print(f"Running sweep for tau={tau}, mu={mu}")

        # Create and run the model
        model = OpinionDynamicsModel(N=N, tau=tau, mu=mu, seed=seed)
        model.run_simulation(steps=steps)

        # Retrieve final number of clusters
        final_data = model.datacollector.get_model_vars_dataframe()
        num_clusters = final_data['NumClusters'].iloc[-1]

        # Record the cluster locations
        final_opinions = [agent.opinion for agent in model.agents]
        cluster_centers = []

        while final_opinions:
            center = final_opinions.pop(0)
            cluster = [center] + [op for op in final_opinions if abs(op - center) < CONFIG["threshold"]]
            final_opinions = [op for op in final_opinions if abs(op - center) >= CONFIG["threshold"]]
            cluster_centers.append(np.mean(cluster))

        results.append({
            "seed": seed,
            "tau": tau,
            "mu": mu,
            "num_clusters": num_clusters,
            "cluster_centers": cluster_centers
        })

    # Convert results to a DataFrame for easy analysis
    return pd.DataFrame(results)

def single_simulation(tau, mu, steps=CONFIG["steps"], N=CONFIG["N"], seed=CONFIG["seed"]):
    """
    Run a single simulation with specified tau and mu values.
    Returns: The model instance after the simulation
    """
    print(f"Running single simulation for tau={tau}, mu={mu}")
    model = OpinionDynamicsModel(N=N, tau=tau, mu=mu, seed=seed)
    model.run_simulation(steps=steps)
    return model

def main():
    print(f"Mesa version: {mesa.__version__}")
    
    # Determine mode of operation: sweep or single simulation
    if CONFIG["mode"] == "sweep":
        # Perform parameter sweep
        sweep_results = parameter_sweep(CONFIG["tau_values"], CONFIG["mu_values"])
        #print(sweep_results)
        # Save results to CSV
        sweep_results.to_csv(os.path.join(os.path.dirname(__file__), "parameter_sweep_results.csv"), index=False)
        print(f"The results of the parameter sweep are saved in the same folder as this file")

    elif CONFIG["mode"] == "single":
        # Run a single simulation
        model = single_simulation(CONFIG["tau"], CONFIG["mu"])
        # Visualize results
        plot_results(model)
        # Retrieve and print final simulation statistics
        final_data = model.datacollector.get_model_vars_dataframe()
        print("\nFinal Simulation Statistics:")
        print(f"Final Mean Opinion: {final_data['MeanOpinion'].iloc[-1]:.4f}")
        print(f"Final Opinion Standard Deviation: {final_data['OpinionStdDev'].iloc[-1]:.4f}")
        print(f"Final Number of Clusters: {final_data['NumClusters'].iloc[-1]}")

if __name__ == "__main__":
    main()
