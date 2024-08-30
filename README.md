# Reinforcement Learning for Traffic Engineering

## Description
Research Project with the goal of developing a reinforcement learning agent that can intelligently distribute network traffic across multiple tunnels, optimizing for user preference, cost, and network performance.

## Short description of each file
Main files:
actor_critic.py - Contains the actor and critic neural networks
ddpg.py - Contains all necessary DDPG algorithm code (only constructor and "act" is used in basic (current) implementation)
Testbed_testing.ipynb - Code to train agent in remote machine test setup described below, also the cleanest version of "final code"
Offline_training.ipynb - Similar to previous, but training agent with random network performance mainly to observe basic functionality in response to user preference and cost. Was initially used to test DDPG, but mostly commented out or removed when debugging.

Shell scripts for testbed setup:
start_iperf_servers.sh - Script to start the 4 iperf servers bound to 4 different tunnel IPs (needs to be set up prior) 
Usage: ./start_iperf_servers.sh
initial-tc-setup.sh - Script to set up qdiscs on third router node to enable network manipulation via netem
Usage: ./initial-tc-setup.sh
apply-netem.sh - Script to apply netem manipulations
Usage to enable one-way delay: sudo ./apply-netem.sh ${FLOW_ID} delay 100ms
Usage to disable netem: sudo ./apply-netem.sh ${FLOW_ID}
make-ecmp-route.sh - Creates 4 ecmp routes through specified tunnels, initializing all with weight 1 (used in initial test setup, scrapped for current)
Usage: ./make-ecmp-route.sh
adjust-ecmp-weights.sh - Script to change the weights of the ecmp routes
Usage: ./adjust-ecmp-weights.sh ${weight 1} ${weight 2} ${weight 3} ${weight 4}

Experimental files (no longer used and mostly disorganized):
RL_Actor_Critic_manual_experiments - Used to better understand math behind updating neural networks via gradient descent, created and updated neural networks with matrices. 
RL_AC_step_update.ipynb - Simple Actor Critic Algorithm implementation with poor results
RL_A2C_episodic_updates.ipynb - Attempted implementation of Advantage Actor Critic unsuccessfully, realized it wasn't meant for this problem. 
traffic_client.py - Fully functional, was used to generate iperf traffic on client node and parse its output to send metrics via websocket to router node (where the RL agent existed). IPerf didn't output the necessary metrics though, so was scrapped.

## Testbed Setup Details


## Authors and acknowledgment
Primarily authored by Neil Kachappilly, with contributions from Nathan Moos and Altanai Bisht.