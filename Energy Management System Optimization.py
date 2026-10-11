import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

# =========================================================
# REAL DATA FROM TRANSELECTRICA + OPCOM
# =========================================================

pret_ora = np.array([
    693, 602, 616, 622, 662, 738, 667, 550,
    206, 84, 126, 146, 153, 191, 247, 381,
    572, 727, 831, 1006, 797, 703, 754, 802
])

consum_ora = np.array([
    5598, 5520, 5568, 5474, 5684, 5968,
    6404, 7115, 7262, 7106, 6646, 6718,
    6534, 6432, 5598, 5961, 6544, 7073,
    7677, 7735, 7419, 6988, 6337, 5862
])

foto_ora = np.array([
    0, 0, 0, 0, 0, 0,
    10, 180, 457, 794, 957, 919,
    1037, 969, 1562, 1048, 500, 113,
    0, 0, 0, 0, 0, 0
])

eolian_ora = np.array([
    698, 714, 953, 1058, 1200, 1230,
    1204, 1037, 1001, 1053, 856, 769,
    848, 1150, 1319, 1211, 1024, 769,
    594, 602, 593, 685, 627, 651
])

pret_ora = np.tile(pret_ora, 2)
consum_ora = np.tile(consum_ora, 2)
foto_ora = np.tile(foto_ora, 2)
eolian_ora = np.tile(eolian_ora, 2)

# =========================================================
# 48-HOUR SCENARIO WITH A 5-MINUTE TIME STEP
# =========================================================

# Simulation time step (hours)
dt = 1/12

T = int(48 / dt)  # 48 hours with time step dt

ore = np.arange(1, T + 1)

ore_base = np.arange(48)
ore_interp = ore * dt

# =========================================================
# DATA INTERPOLATION
# =========================================================

f_pret = interp1d(ore_base, pret_ora, kind='linear', fill_value="extrapolate")
f_cons = interp1d(ore_base, consum_ora, kind='linear', fill_value="extrapolate")
f_ren  = interp1d(ore_base, foto_ora + eolian_ora, kind='linear', fill_value="extrapolate")

C = f_pret(ore_interp)
D = f_cons(ore_interp)
P_regen = f_ren(ore_interp)

# Surplus electricity selling price, reduced relative to the purchase price
C_vanzare = 0.9 * C

# =========================================================
# ENERGY MANAGEMENT SYSTEM MODEL
# =========================================================

model = gp.Model("SME_2_zile")

# =========================================================
# DYNAMIC PARAMETERS
# =========================================================

# Nuclear power: slow response (tau = 30 min)
alpha_nuc = np.exp(-dt / (1/2))

# Hydropower: fast response (tau = 10 min)
alpha_hid = np.exp(-dt / (1/6))

# Nuclear and hydropower operating costs (per MWh generated)
c_nuc = 150
c_hid = 50

# Cost coefficients penalizing changes in nuclear and hydropower setpoints
c_delta_nuc = 1000
c_delta_hid = 300

# Maximum total hydropower energy over the 48-hour horizon (limited water supply)
E_hid_max = 100000  # Equivalent to a constant output of approximately 2083 MW over 48 hours

# Battery charging and discharging efficiencies
beta_inc = 0.95
beta_des = 0.95

# Battery degradation cost and charging/discharging power variation limit
c_bat = 1
R_bat = 300 * dt  # Power variation limit corresponding to 300 MW/h

# Battery self-discharge rate (0.003% per hour)
self_dis = 0.00003

# =========================================================
# DECISION VARIABLES
# =========================================================

# Setpoints
u_nuc = model.addVars(T, lb=900, ub=1800, name="u_nuc")
u_hid = model.addVars(T, lb=250, ub=2500, name="u_hid")

# Nuclear and hydropower output
P_nuc = model.addVars(T, lb=900, ub=1800, name="P_nuc")
P_hid = model.addVars(T, lb=250, ub=2500, name="P_hid")

# Variables penalizing rapid changes in nuclear and hydropower setpoints
delta_nuc = model.addVars(T, lb=0, name="delta_nuc")
delta_hid = model.addVars(T, lb=0, name="delta_hid")

# Battery
P_inc = model.addVars(T, lb=0, ub=500, name="P_inc")
P_des = model.addVars(T, lb=0, ub=500, name="P_des")  # Maximum charging/discharging power: 500 MW

# The battery cannot charge and discharge simultaneously
# 1: charging
# 0: discharging

stare_baterie = model.addVars(T, vtype=GRB.BINARY, name="stare_baterie")

SoC = model.addVars(T + 1, lb=100, ub=2500, name="SoC")

# Grid imports and exports
P_import = model.addVars(T, lb=0, name="P_import")
P_export = model.addVars(T, lb=0, name="P_export")

# The grid cannot import and export electricity simultaneously
stare_retea = model.addVars(T, vtype=GRB.BINARY, name="stare_retea")

# =========================================================
# INITIAL CONDITIONS
# =========================================================

# Initial nuclear power output
model.addConstr(P_nuc[0] == 1300)

# Initial hydropower output
model.addConstr(P_hid[0] == 1200)

# Initial battery state of charge
model.addConstr(SoC[0] == 500)

# =========================================================
# SYSTEM DYNAMICS
# =========================================================

for t in range(T - 1):

    # =====================================================
    # NUCLEAR POWER DYNAMICS
    # =====================================================

    # Slow response

    model.addConstr(P_nuc[t + 1] == alpha_nuc * P_nuc[t] + (1 - alpha_nuc) * u_nuc[t])

    model.addConstr(delta_nuc[t] >= u_nuc[t+1] - u_nuc[t])
    model.addConstr(delta_nuc[t] >= u_nuc[t] - u_nuc[t+1])

    # =====================================================
    # HYDROPOWER DYNAMICS
    # =====================================================

    # Fast response

    model.addConstr(P_hid[t + 1] == alpha_hid * P_hid[t] + (1 - alpha_hid) * u_hid[t])

    model.addConstr(delta_hid[t] >= u_hid[t+1] - u_hid[t])
    model.addConstr(delta_hid[t] >= u_hid[t] - u_hid[t+1])

# Maximum total hydropower energy over the simulation horizon
model.addConstr(gp.quicksum(P_hid[t] * dt for t in range(T)) <= E_hid_max)

# =========================================================
# BATTERY DYNAMICS
# =========================================================

for t in range(T):

    # The battery can either charge or discharge
    model.addConstr(P_inc[t] <= 500 * stare_baterie[t])
    model.addConstr(P_des[t] <= 500 * (1 - stare_baterie[t]))

    model.addConstr(SoC[t + 1] == (1 - self_dis * dt) * SoC[t] + (beta_inc * P_inc[t] - P_des[t] / beta_des) *dt)

for t in range(T - 1):

    # Limit on charging/discharging power variations
    model.addConstr(P_inc[t+1] - P_inc[t] <= R_bat)
    model.addConstr(P_inc[t] - P_inc[t+1] <= R_bat)

    model.addConstr(P_des[t+1] - P_des[t] <= R_bat)
    model.addConstr(P_des[t] - P_des[t+1] <= R_bat)

# Final battery state of charge
model.addConstr(SoC[T] >= SoC[0])

# =========================================================
# ENERGY BALANCE
# =========================================================

for t in range(T):

    model.addConstr(P_nuc[t] + P_hid[t] + P_regen[t] + P_des[t] + P_import[t] == D[t] + P_inc[t] + P_export[t])

    # =====================================================
    # GRID IMPORT/EXPORT CONSTRAINTS
    # =====================================================

    model.addConstr(P_import[t] <= 10000 * stare_retea[t])
    model.addConstr(P_export[t] <= 10000 * (1 - stare_retea[t]))

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
#
# Minimize grid electricity import costs and generation operating costs:
#
# J = sum(total import cost + nuclear cost + hydropower cost
#         + battery cost - total export revenue)
#
# The expected outcomes are:
# - The system prioritizes its own generation sources
# - The battery discharges during periods of high electricity prices
# - Grid imports are minimized

cost_total = gp.quicksum(dt * (C[t] * P_import[t] + c_nuc * P_nuc[t] + c_hid * P_hid[t] + c_bat * (P_inc[t] + P_des[t])
                               - C_vanzare[t] * P_export[t]) + c_delta_nuc * delta_nuc[t] + c_delta_hid * delta_hid[t] for t in range(T))

model.setObjective(cost_total, GRB.MINIMIZE)

# =========================================================
# OPTIMIZATION
# =========================================================

model.setParam("Presolve", 0)
model.optimize()

# =====================================================
# CONSOLE OUTPUT
# =====================================================

total_import = sum(P_import[t].X * dt for t in range(T))
total_export = sum(P_export[t].X * dt for t in range(T))
utilizare_baterie = sum((P_inc[t].X + P_des[t].X) * dt for t in range(T))
E_inc = sum(P_inc[t].X * dt for t in range(T))
E_des = sum(P_des[t].X * dt for t in range(T))

print("\n===================================")
print(f"Total imports:        {total_import:,.2f} MWh")
print(f"Total exports:        {total_export:,.2f} MWh")
print(f"Battery throughput:   {utilizare_baterie:,.2f} MWh \n({E_inc:,.2f} MWh charged + {E_des:,.2f} MWh discharged)")
print(f"\nTotal cost: {model.objVal:,.2f} RON")
print("===================================")

# =========================================================
# VISUALIZATION
# =========================================================

if model.status == GRB.OPTIMAL:

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # X-axis tick positions
    ore_gradatii = [1/dt, 6/dt, 12/dt, 18/dt, 24/dt, 30/dt, 36/dt, 42/dt, 48/dt]
    etichete = ['1', '6', '12', '18', '24', '6', '12', '18', '24']

    for ax in axes:
        ax.set_xlim(1, 48)
        ax.set_xticks(ore_gradatii)
        ax.set_xticklabels(etichete)
        ax.grid(True, linestyle='--', alpha=0.6)

    # =====================================================
    # CONSUMPTION VS. GENERATION PLOT
    # =====================================================

    productie_totala = [P_nuc[t].X + P_hid[t].X + P_regen[t] for t in range(T)]

    # Consumption curve
    axes[0].plot(ore, D, color='red', linewidth=2, label='Consumption')

    # Fill area under the consumption curve
    axes[0].fill_between(ore, D, color='red', alpha=0.3)

    # Generation curve
    axes[0].plot(ore, productie_totala, color='blue', linewidth=2, label='Generation')

    # Fill area under the generation curve
    axes[0].fill_between(ore, productie_totala, color='blue', alpha=0.3)

    axes[0].set_title("Consumption vs. Generation", fontsize=11, fontweight='bold')

    axes[0].set_ylabel("Power [MW]")
    axes[0].legend(framealpha=0.65)

    # =====================================================
    # POWER SOURCE DYNAMICS PLOT
    # =====================================================

    axes[1].plot(ore, [P_nuc[t].X for t in range(T)], label='Nuclear', linewidth=2)

    axes[1].plot(ore, [P_hid[t].X for t in range(T)], label='Hydropower', linewidth=2)

    axes[1].plot(ore, P_regen, label='PV + Wind', linewidth=2)

    axes[1].set_title("Power Source Dynamics", fontsize=11, fontweight='bold')

    axes[1].set_ylabel("Power [MW]")
    axes[1].legend(framealpha=0.65)

    # =====================================================
    # BATTERY STATE OF CHARGE (SoC) PLOT
    # =====================================================

    axes[2].plot(np.arange(0, T + 1), [SoC[t].X for t in range(T + 1)], color='purple', linewidth=2)

    for t in range(T):
        if P_inc[t].X > 1e-3:
            axes[2].axvspan(t, t+1, color='lightgreen', alpha=0.25)

        if P_des[t].X > 1e-3:
            axes[2].axvspan(t, t+1, color='lightcoral', alpha=0.25)
        else:
            axes[2].axvspan(t, t+1, color='lightgray', alpha=0.25)

    axes[2].set_title("Battery State of Charge", fontsize=11, fontweight='bold')

    axes[2].set_ylabel("Energy [MWh]")

    legenda = [
    mpatches.Patch(color='lightgreen', alpha=0.3, label='Charging'),
    mpatches.Patch(color='lightcoral', alpha=0.3, label='Discharging'),
    mpatches.Patch(color='purple', label='SoC')]

    axes[2].legend(handles=legenda, loc='upper left', framealpha=0.65)

    # =====================================================
    # GRID IMPORT/EXPORT AND ELECTRICITY PRICE PLOT
    # =====================================================

    cost_import = [(C[t] * P_import[t].X * dt) / 1000 for t in range(T)]

    profit_export = [-(C_vanzare[t] * P_export[t].X * dt) / 1000 for t in range(T)]

    ax1 = axes[3]

    ax1.bar(ore, cost_import, color='darkred', alpha=0.7, label='Imports')
    ax1.bar(ore, profit_export, color='darkgreen', alpha=0.7, label='Exports')

    ax1.axhline(0, color='black', linewidth=1)

    ax1.set_ylabel("Cost [thousand RON]")

    ax1.set_title("Energy Imports/Exports and Electricity Prices", fontsize=11, fontweight='bold')

    # Secondary axis for electricity price
    ax2 = ax1.twinx()

    ax2.plot(ore, C, color='darkorange', linewidth=2, label='Electricity Price')

    ax2.set_ylabel("Price [RON/MWh]")
    ax2.set_ylim(bottom=0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.0525, 0.237), frameon=True, framealpha=0.65)

    ax1.set_xlabel("Hour", labelpad=-1)

    plt.tight_layout()
    plt.show()
