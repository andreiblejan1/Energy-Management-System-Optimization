# Energy-Management-System-Optimization
Python-based optimization of a hybrid energy system integrating nuclear, hydroelectric and renewable generation, as well as battery storage and grid import/export with dynamic prices.
Formulates a Mixed-Integer Linear Programming (MILP) model solved using Gurobi, minimizing operating costs using real electricity market and national power system data from OPCOM and Transelectrica.
Simulates a 48-hour horizon with 5-minute time steps, incorporating generator dynamics, battery state-of-charge constraints, energy balance, and import/export decisions.
