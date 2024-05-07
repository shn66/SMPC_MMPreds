# Stochastic MPC with Multimodal Predictions
Implementation of stochastic MPC using mode/disturbance feedback policies to handle multimodal predictions of other agents.

The predictions models were trained using the code found at https://github.com/govvijaycal/confidence_aware_predictions.

### Overall Setup

First follow instractions in envs/setup.txt to setup the conda environment.

### Setup for Carla/Simulation

1. Install CARLA (I used 0.9.10).
2. See envs/setup.txt to set up Carla and pytope.

---

## Carla Simulation

The main script here is run_all_scenarios.py, which looks up the scenarios defined in carla/scenarios/ and executes all specified initial conditions.  Scenarios are defined by providing an intersection layout (csv) and a json specificying all agents and configuration parameters.  The closed loop trajectories and a video can be saved per execution of a scenario.  See scenarios/run_interesection_scenario.py for further details.

The specific control policies are defined in carla/policies.  These include a collision-avoiding MPC lanekeeping agent (used for the target vehicle and considering the multimodal predictions) and three SMPC formulations : (1) open-loop, (2) fixed-risk formulation and (3) variable-risk formulation. There is also an SMPC implementation that uses optmization-based collision avoidance.
