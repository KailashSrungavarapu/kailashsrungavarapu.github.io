"""
Turbocharged Diesel Dual-Cycle (Air-Standard) Simulation
-----------------------------------------------------------
Models a heavy-duty truck diesel engine cycle as an air-standard
dual cycle (Diesel cycle modified to include both constant-volume
AND constant-pressure heat addition, which better matches real
diesel combustion than the idealized pure-Diesel cycle).

Process path (state points 1-2-3-4-5-1):
  1 -> 2 : Isentropic compression (compression ratio r)
  2 -> 3 : Constant-volume heat addition (pressure ratio rp) -- rapid
           premixed/kinetically-controlled combustion phase
  3 -> 4 : Constant-pressure heat addition (cutoff ratio rc) -- slower
           diffusion-controlled combustion phase
  4 -> 5 : Isentropic expansion
  5 -> 1 : Constant-volume heat rejection (exhaust blowdown)

A turbocharger is modeled simply as a boost stage that raises the
intake pressure/density before state 1, increasing the mass of air
(and therefore fuel) that can be burned per cycle -- this is the
primary mechanism by which turbocharging increases specific power
output in real diesel engines.

Author: Kailash Srungavarapu
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Air-standard properties (cold-air assumption, representative of
# common engineering-textbook engine cycle analysis)
# ---------------------------------------------------------------
R = 0.287         # kJ/(kg.K)  specific gas constant for air
cv = 0.718        # kJ/(kg.K) at ~300-900K representative average
cp = 1.005        # kJ/(kg.K)
gamma = cp / cv   # ~1.4

# ---------------------------------------------------------------
# Engine / cycle input parameters (representative of a modern
# heavy-duty turbocharged truck diesel engine)
# ---------------------------------------------------------------
r   = 17.0        # compression ratio (typical HD diesel: 15-18)
rc  = 1.6         # cutoff ratio (V4/V3) - constant pressure phase
rp  = 1.25        # pressure ratio (P3/P2) - constant volume phase

# Ambient / intake conditions
P0 = 100.0        # kPa, ambient pressure
T0 = 300.0        # K, ambient temperature

# Turbocharger boost model (simple isentropic compressor + fixed
# pressure ratio, matched to typical HD diesel boost levels)
boost_pressure_ratio = 2.2     # typical single-stage turbo PR
compressor_isentropic_eff = 0.75

def turbo_boost(P0, T0, PR, eta_c):
    """Simple isentropic-efficiency compressor model for the turbo."""
    P1 = P0 * PR
    T1_ideal = T0 * PR ** ((gamma - 1) / gamma)
    T1_actual = T0 + (T1_ideal - T0) / eta_c
    return P1, T1_actual

# ---------------------------------------------------------------
# Cycle calculation
# ---------------------------------------------------------------
def compute_dual_cycle(P1, T1, r, rp, rc):
    """Compute all 5 state points of the air-standard dual cycle."""
    V1 = 1.0                          # normalized specific volume (m^3/kg, arbitrary reference)
    V2 = V1 / r

    # State 1: after intake (post-turbo boost, at BDC)
    state1 = dict(P=P1, T=T1, V=V1)

    # 1->2: isentropic compression
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    state2 = dict(P=P2, T=T2, V=V2)

    # 2->3: constant-volume heat addition
    P3 = P2 * rp
    T3 = T2 * rp
    V3 = V2
    state3 = dict(P=P3, T=T3, V=V3)

    # 3->4: constant-pressure heat addition
    P4 = P3
    V4 = V3 * rc
    T4 = T3 * rc
    state4 = dict(P=P4, T=T4, V=V4)

    # 4->5: isentropic expansion back to V1
    V5 = V1
    T5 = T4 * (V4 / V5) ** (gamma - 1)
    P5 = P4 * (V4 / V5) ** gamma
    state5 = dict(P=P5, T=T5, V=V5)

    return [state1, state2, state3, state4, state5]

def cycle_performance(states, r, rp, rc):
    T1, T2, T3, T4, T5 = [s['T'] for s in states]

    q_in_v = cv * (T3 - T2)     # constant-volume heat addition
    q_in_p = cp * (T4 - T3)     # constant-pressure heat addition
    q_in = q_in_v + q_in_p

    q_out = cv * (T5 - T1)      # constant-volume heat rejection

    w_net = q_in - q_out
    eta_thermal = w_net / q_in

    # Analytical dual-cycle efficiency formula (cross-check)
    eta_formula = 1 - (1 / (r ** (gamma - 1))) * (
        (rp * rc ** gamma - 1) / ((rp - 1) + gamma * rp * (rc - 1))
    )

    return dict(q_in=q_in, q_out=q_out, w_net=w_net,
                eta_thermal=eta_thermal, eta_formula=eta_formula)

# ---------------------------------------------------------------
# Run baseline case
# ---------------------------------------------------------------
P1, T1 = turbo_boost(P0, T0, boost_pressure_ratio, compressor_isentropic_eff)
states = compute_dual_cycle(P1, T1, r, rp, rc)
perf = cycle_performance(states, r, rp, rc)

print("=" * 60)
print("TURBOCHARGED DIESEL DUAL-CYCLE - BASELINE RESULTS")
print("=" * 60)
print(f"Ambient:           P0 = {P0:.1f} kPa,  T0 = {T0:.1f} K")
print(f"Turbo boost PR:     {boost_pressure_ratio:.2f}  (eta_c = {compressor_isentropic_eff:.2f})")
print(f"Post-turbo intake:  P1 = {P1:.1f} kPa,  T1 = {T1:.1f} K")
print("-" * 60)
labels = ["1 (intake, post-turbo)", "2 (end compression)",
          "3 (end const-V heat add)", "4 (end const-P heat add)",
          "5 (end expansion)"]
for lbl, s in zip(labels, states):
    print(f"State {lbl:28s}  P = {s['P']:8.1f} kPa   T = {s['T']:7.1f} K   V = {s['V']:.4f}")
print("-" * 60)
print(f"Heat in (q_in):      {perf['q_in']:.2f} kJ/kg")
print(f"Heat out (q_out):    {perf['q_out']:.2f} kJ/kg")
print(f"Net work (w_net):    {perf['w_net']:.2f} kJ/kg")
print(f"Thermal efficiency:  {perf['eta_thermal']*100:.2f} %  (energy-balance calc)")
print(f"Thermal efficiency:  {perf['eta_formula']*100:.2f} %  (closed-form formula, cross-check)")
print("=" * 60)

# ---------------------------------------------------------------
# Build full P-V path (with polytropic curves for compression/
# expansion legs, not just straight lines between state points)
# ---------------------------------------------------------------
def pv_path(states):
    s1, s2, s3, s4, s5 = states
    V_12 = np.linspace(s1['V'], s2['V'], 100)
    P_12 = s1['P'] * (s1['V'] / V_12) ** gamma          # isentropic compression

    V_23 = np.array([s2['V'], s3['V']])
    P_23 = np.array([s2['P'], s3['P']])                  # constant volume

    V_34 = np.linspace(s3['V'], s4['V'], 100)
    P_34 = np.full_like(V_34, s3['P'])                    # constant pressure

    V_45 = np.linspace(s4['V'], s5['V'], 100)
    P_45 = s4['P'] * (s4['V'] / V_45) ** gamma          # isentropic expansion

    V_51 = np.array([s5['V'], s1['V']])
    P_51 = np.array([s5['P'], s1['P']])                  # constant volume

    V = np.concatenate([V_12, V_23, V_34, V_45, V_51])
    P = np.concatenate([P_12, P_23, P_34, P_45, P_51])
    return V, P

V_path, P_path = pv_path(states)

# ---------------------------------------------------------------
# Plot 1: P-V diagram
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(V_path, P_path, color='#2E5C8A', linewidth=2)
ax.fill(V_path, P_path, color='#2E5C8A', alpha=0.08)

for i, s in enumerate(states, start=1):
    ax.plot(s['V'], s['P'], 'o', color='#C0722C', markersize=6, zorder=5)
    ax.annotate(str(i), (s['V'], s['P']), textcoords="offset points",
                xytext=(8, 8), fontsize=11, fontweight='bold', color='#151D24')

ax.set_xlabel('Specific Volume, V (m³/kg, normalized)', fontsize=11)
ax.set_ylabel('Pressure, P (kPa)', fontsize=11)
ax.set_title('Turbocharged Diesel Dual-Cycle — P-V Diagram', fontsize=13, fontweight='bold')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('dual_cycle_pv.png', dpi=150, facecolor='white')
print("Saved dual_cycle_pv.png")

# ---------------------------------------------------------------
# Plot 2: T-s diagram (entropy computed relative to state 1)
# ---------------------------------------------------------------
def entropy_path(states):
    s1, s2, s3, s4, s5 = states
    s_ref = 0.0

    # 1->2 isentropic: ds = 0
    T_12 = np.linspace(s1['T'], s2['T'], 100)
    s_12 = np.full_like(T_12, s_ref)

    # 2->3 constant volume: ds = cv * ln(T3/T2)
    T_23 = np.linspace(s2['T'], s3['T'], 100)
    s_23 = s_ref + cv * np.log(T_23 / s2['T'])

    # 3->4 constant pressure: ds = cp * ln(T4/T3)
    s3_val = s_23[-1]
    T_34 = np.linspace(s3['T'], s4['T'], 100)
    s_34 = s3_val + cp * np.log(T_34 / s3['T'])

    # 4->5 isentropic: ds = 0
    s4_val = s_34[-1]
    T_45 = np.linspace(s4['T'], s5['T'], 100)
    s_45 = np.full_like(T_45, s4_val)

    # 5->1 constant volume: ds = cv * ln(T1/T5)
    s5_val = s_45[-1]
    T_51 = np.linspace(s5['T'], s1['T'], 100)
    s_51 = s5_val + cv * np.log(T_51 / s5['T'])

    T = np.concatenate([T_12, T_23, T_34, T_45, T_51])
    s = np.concatenate([s_12, s_23, s_34, s_45, s_51])
    s_states = [s_ref, s_ref, s3_val, s4_val, s5_val]
    return s, T, s_states

s_path, T_path, s_states = entropy_path(states)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(s_path, T_path, color='#2E5C8A', linewidth=2)

for i, (s, st) in enumerate(zip(s_states, states), start=1):
    ax.plot(s, st['T'], 'o', color='#C0722C', markersize=6, zorder=5)
    ax.annotate(str(i), (s, st['T']), textcoords="offset points",
                xytext=(8, 8), fontsize=11, fontweight='bold', color='#151D24')

ax.set_xlabel('Specific Entropy, s - s1 (kJ/kg.K)', fontsize=11)
ax.set_ylabel('Temperature, T (K)', fontsize=11)
ax.set_title('Turbocharged Diesel Dual-Cycle — T-s Diagram', fontsize=13, fontweight='bold')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('dual_cycle_ts.png', dpi=150, facecolor='white')
print("Saved dual_cycle_ts.png")

# ---------------------------------------------------------------
# Plot 3: Efficiency vs compression ratio, for a few cutoff ratios
# ---------------------------------------------------------------
def eta_dual(r, rp, rc):
    return 1 - (1 / (r ** (gamma - 1))) * (
        (rp * rc ** gamma - 1) / ((rp - 1) + gamma * rp * (rc - 1))
    )

r_sweep = np.linspace(10, 24, 100)
fig, ax = plt.subplots(figsize=(7.5, 6))
colors = ['#2E5C8A', '#C0722C', '#4A5560']
for rc_val, color in zip([1.3, 1.6, 2.0], colors):
    eta_sweep = eta_dual(r_sweep, rp, rc_val) * 100
    ax.plot(r_sweep, eta_sweep, color=color, linewidth=2,
             label=f'Cutoff ratio rc = {rc_val}')

ax.axvline(r, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.text(r + 0.2, ax.get_ylim()[0] + 2, f'Baseline r = {r}', fontsize=9, color='gray')

ax.set_xlabel('Compression Ratio, r', fontsize=11)
ax.set_ylabel('Thermal Efficiency, η (%)', fontsize=11)
ax.set_title('Dual-Cycle Efficiency vs Compression Ratio', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('dual_cycle_efficiency_sweep.png', dpi=150, facecolor='white')
print("Saved dual_cycle_efficiency_sweep.png")

print("\nAll plots generated successfully.")
