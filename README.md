# Agent_MPP

This notebook contains a reinforcement learning agent designed to learn the optimal strategy for Merton's Portfolio Problem, a classic problem in finance theory. The problem involves a continuous-time, stochastic, intertemporal optimization problem concerning how much an individual should consume and invest in risky assets in order to maximize their lifetime utility of consumption.

## Code Structure

The code is divided into two main parts:

### 1. Reinforcement Learning Agent (PogChamp Class)

The first part of the code defines a reinforcement learning agent, represented by the `PogChamp` class. The agent has the following key attributes and methods:

- `state`: A state space vector representing the time as a percentage of time left in the episode.
- `actions`: An action space vector where the first element represents the percentage of wealth to invest in risky assets, and the second element represents the percentage of wealth to consume.
- `q_table`: A 2-dimensional table to store the values of state-action pairs.
- `action_step`: A method that performs an action (investment and consumption), logs the action, calculates the reward for the action, and updates the wealth based on the action taken.
- `get_dW`: A method that calculates the change in wealth based on the action taken and the stochastic behavior of the risky asset.
- `get_reward`: A method that calculates the utility of wealth or consumption based on the Constant Relative Risk Aversion (CRRA) utility function.

### 2. Simulation of Agent's Learning Process

The second part of the code simulates the learning process of the agent over multiple episodes. The agent learns by exploring and exploiting the action space based on an epsilon-greedy strategy, updating its Q-table, and attempting to maximize the cumulative reward.

## Results

The code produces plots visualizing the learning process, including:

- The learning process
- The total return over episodes.
- The frequency of different levels of investment in risky assets.
- A sample trajectory of consumption over time.
- A visualization of the Q-table.

It also prints the total amount of money consumed and the total amount of money left over at the end of the simulation.

# Optimal agent
## Merton's Portfolio Problem - Analytical Solution

This notebook contains the implementation of the analytical solution of Merton's portfolio problem.

Merton's portfolio problem is an optimization problem in finance in which an investor decides how much to consume and invest in a risky and riskless asset to maximize his expected utility of consumption over a finite horizon. This project uses a simulation to derive the solution.

# MertonRL 

This is a pdf of the report and analysis of the data.



