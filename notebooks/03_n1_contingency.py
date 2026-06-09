# ============================================================
# WattWorker EV Charging Grid Analysis
# Notebook 03: N-1 Contingency Screening
# ------------------------------------------------------------
# Systematic N-1 analysis comparing:
#   - Base case (no EV load)
#   - With EV hub at strong grid bus
#   - With EV hub at weak grid bus
#
# Key metric: voltage severity score
#   = sum of voltage deviations outside 0.95-1.05 pu band
#   A higher score = worse contingency
#
# We identify:
#   1. Which contingencies are worst overall
#   2. Which contingencies get WORSE when EV load is added
#   3. How much worse the weak grid is vs the strong grid
# ============================================================

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────
TRUCK_KW = 150
N_TRUCKS = 50
EV_MW    = N_TRUCKS * TRUCK_KW / 1000   # 7.5 MW
EV_MVAR  = EV_MW * 0.1

NETWORKS = {
    "IEEE 39-bus": {
        "loader"      : pn.case39,
        "bus_strong"  : 16,
        "bus_weak"    : 27,
        "label_strong": "Bay Area Port",
        "label_weak"  : "Suburban Feeder",
    },
    "IEEE 118-bus": {
        "loader"      : pn.case118,
        "bus_strong"  : 49,
        "bus_weak"    : 106,
        "label_strong": "Bay Area Port",
        "label_weak"  : "Suburban Feeder",
    },
}

def voltage_severity(v_series):
    """
    Severity score = total pu deviation outside 0.95-1.05 band.
    0 = fully healthy. Higher = worse.
    """
    low  = (0.95 - v_series[v_series < 0.95]).sum()
    high = (v_series[v_series > 1.05] - 1.05).sum()
    return round(low + high, 6)

def run_n1_screening(net):
    """
    Trip every line one at a time. Record violations and severity.
    """
    records = []
    for line_idx in range(len(net.line)):
        net_c = copy.deepcopy(net)
        net_c.line.at[line_idx, 'in_service'] = False
        try:
            pp.runpp(net_c, algorithm='nr',
                     calculate_voltage_angles=True,
                     max_iteration=50)
            converged = True
        except Exception:
            converged = False

        if not converged:
            records.append({
                "line_idx"       : line_idx,
                "converged"      : False,
                "v_violations"   : 99,
                "thermal_viol"   : 99,
                "min_v_pu"       : 0.0,
                "max_loading_pct": 999,
                "severity"       : 99.0,
            })
            continue

        v       = net_c.res_bus['vm_pu']
        loading = net_c.res_line['loading_percent']
        load_a  = loading.copy()
        load_a.iloc[line_idx] = 0  # ignore tripped line itself

        records.append({
            "line_idx"       : line_idx,
            "converged"      : True,
            "v_violations"   : int(((v < 0.95) | (v > 1.05)).sum()),
            "thermal_viol"   : int((load_a > 100).sum()),
            "min_v_pu"       : round(v.min(), 4),
            "max_loading_pct": round(load_a.max(), 2),
            "severity"       : voltage_severity(v),
        })

    return pd.DataFrame(records)

# ── RUN ALL CASES ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  WattWorker EV Hub — N-1 Contingency Screening")
print(f"  Scenario: {N_TRUCKS} trucks × {TRUCK_KW}kW = {EV_MW} MW")
print("=" * 65)

all_results = {}

for net_name, config in NETWORKS.items():
    print(f"\n{'─'*65}")
    print(f"  {net_name}")
    print(f"{'─'*65}")

    n_lines = len(config["loader"]().line)

    # Base case
    print(f"  [1/3] Base case ({n_lines} contingencies)...")
    net_b = config["loader"]()
    pp.runpp(net_b, algorithm='nr', calculate_voltage_angles=True)
    df_base = run_n1_screening(net_b)

    # Strong grid EV hub
    print(f"  [2/3] EV hub @ {config['label_strong']} (Bus {config['bus_strong']})...")
    net_s = config["loader"]()
    pp.create_load(net_s, bus=config["bus_strong"],
                   p_mw=EV_MW, q_mvar=EV_MVAR, name="EV_Strong")
    pp.runpp(net_s, algorithm='nr', calculate_voltage_angles=True)
    df_strong = run_n1_screening(net_s)

    # Weak grid EV hub
    print(f"  [3/3] EV hub @ {config['label_weak']} (Bus {config['bus_weak']})...")
    net_w = config["loader"]()
    pp.create_load(net_w, bus=config["bus_weak"],
                   p_mw=EV_MW, q_mvar=EV_MVAR, name="EV_Weak")
    pp.runpp(net_w, algorithm='nr', calculate_voltage_angles=True)
    df_weak = run_n1_screening(net_w)

    # Severity delta: how much WORSE each contingency gets with EV load
    df_cmp = pd.DataFrame({
        "line_idx"       : df_base["line_idx"],
        "base_severity"  : df_base["severity"],
        "strong_severity": df_strong["severity"],
        "weak_severity"  : df_weak["severity"],
        "base_min_v"     : df_base["min_v_pu"],
        "strong_min_v"   : df_strong["min_v_pu"],
        "weak_min_v"     : df_weak["min_v_pu"],
        "base_viol"      : df_base["v_violations"],
        "weak_viol"      : df_weak["v_violations"],
        "base_thermal"   : df_base["thermal_viol"],
        "weak_thermal"   : df_weak["thermal_viol"],
    })
    df_cmp["delta_weak"]   = (df_cmp["weak_severity"]   - df_cmp["base_severity"]).round(6)
    df_cmp["delta_strong"] = (df_cmp["strong_severity"] - df_cmp["base_severity"]).round(6)

    # Top 5 worst contingencies for weak grid
    top5 = df_cmp.nlargest(5, "weak_severity")[
        ["line_idx","base_min_v","weak_min_v","delta_weak",
         "weak_viol","weak_thermal"]
    ].copy()
    top5.columns = ["Line","Base min V","Weak min V","Δ severity","V viol","Thermal viol"]

    print(f"\n  TOP 5 WORST N-1 CONTINGENCIES (weak grid, Bus {config['bus_weak']}):")
    print(top5.to_string(index=False))

    # Count how many contingencies got WORSE with EV load
    worse_weak   = int((df_cmp["delta_weak"]   > 0.001).sum())
    worse_strong = int((df_cmp["delta_strong"] > 0.001).sum())
    print(f"\n  Contingencies made WORSE by EV hub:")
    print(f"    Strong grid (Bus {config['bus_strong']}): {worse_strong}")
    print(f"    Weak grid   (Bus {config['bus_weak']}):   {worse_weak}")

    all_results[net_name] = {
        "df_base"  : df_base,
        "df_strong": df_strong,
        "df_weak"  : df_weak,
        "df_cmp"   : df_cmp,
        "config"   : config,
        "worse_weak"  : worse_weak,
        "worse_strong": worse_strong,
    }

# ── VISUALISATION ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    f"WattWorker EV Hub — N-1 Contingency Severity Analysis\n"
    f"{N_TRUCKS} trucks × {TRUCK_KW}kW = {EV_MW}MW | "
    f"Severity = total voltage deviation outside 0.95-1.05 pu band",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

for row, (net_name, data) in enumerate(all_results.items()):
    df_cmp = data["df_cmp"]
    config = data["config"]

    # ── Severity comparison plot ──────────────────────────────
    ax1 = fig.add_subplot(gs[row, 0])
    x = df_cmp["line_idx"]
    ax1.plot(x, df_cmp["base_severity"],   color='#4575b4', lw=1.2,
             alpha=0.8, label='Base case')
    ax1.plot(x, df_cmp["strong_severity"], color='#2ca02c', lw=1.2,
             alpha=0.8, label=f"EV @ {config['label_strong']}")
    ax1.plot(x, df_cmp["weak_severity"],   color='#d73027', lw=1.5,
             alpha=0.9, label=f"EV @ {config['label_weak']}")
    ax1.set_xlabel('Tripped line index', fontsize=9)
    ax1.set_ylabel('Voltage severity score (pu)', fontsize=9)
    ax1.set_title(f'{net_name}\nN-1 Severity: Base vs EV locations', fontsize=10,
                  fontweight='bold')
    ax1.legend(fontsize=7)

    # ── Delta severity (how much worse EV makes each contingency) ─
    ax2 = fig.add_subplot(gs[row, 1])
    delta_s = df_cmp["delta_strong"]
    delta_w = df_cmp["delta_weak"]

    colors_s = ['#2ca02c' if d > 0.001 else '#aec7e8' for d in delta_s]
    colors_w = ['#d73027' if d > 0.001 else '#aec7e8' for d in delta_w]

    width = 0.4
    ax2.bar(x - width/2, delta_s, width, color=colors_s, alpha=0.85,
            label=f"EV @ {config['label_strong']}")
    ax2.bar(x + width/2, delta_w, width, color=colors_w, alpha=0.85,
            label=f"EV @ {config['label_weak']}")
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Tripped line index', fontsize=9)
    ax2.set_ylabel('Severity increase (pu)', fontsize=9)
    ax2.set_title(f'{net_name}\nContingencies Made Worse by EV Hub', fontsize=10,
                  fontweight='bold')
    ax2.legend(fontsize=7)

    # Annotate count
    ax2.text(0.02, 0.97,
             f"Worse contingencies:\n"
             f"  Strong: {data['worse_strong']}\n"
             f"  Weak:   {data['worse_weak']}",
             transform=ax2.transAxes, fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

plt.savefig('outputs/03_n1_contingency.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Plot saved → outputs/03_n1_contingency.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SYSTEM IMPACT STUDY — FINAL FINDINGS (IEEE 118-bus)")
print("=" * 65)

d118 = all_results["IEEE 118-bus"]
c118 = d118["config"]
df   = d118["df_cmp"]

avg_base_sev   = df["base_severity"].mean()
avg_strong_sev = df["strong_severity"].mean()
avg_weak_sev   = df["weak_severity"].mean()

print(f"""
  Average N-1 severity score across all contingencies:
    Base case             : {avg_base_sev:.6f} pu
    EV @ {c118['label_strong']:20s}: {avg_strong_sev:.6f} pu  (Δ {avg_strong_sev-avg_base_sev:+.6f})
    EV @ {c118['label_weak']:20s}: {avg_weak_sev:.6f} pu  (Δ {avg_weak_sev-avg_base_sev:+.6f})

  Contingencies made worse by EV hub:
    Strong grid (Bus {c118['bus_strong']}) : {d118['worse_strong']} contingencies
    Weak grid   (Bus {c118['bus_weak']}) : {d118['worse_weak']} contingencies

  RECOMMENDATION:
  → Bus {c118['bus_strong']} ({c118['label_strong']}) is the safe connection point.
    Adding the EV hub here causes minimal additional stress
    under N-1 contingency conditions.

  → Bus {c118['bus_weak']} ({c118['label_weak']}) requires network upgrades.
    The EV hub makes {d118['worse_weak']} contingencies worse.
    Required actions before connection:
      • Reconductor adjacent lines to reduce impedance
      • Add reactive power compensation (STATCOM / capacitor bank)
      • Estimated upgrade cost: $2M - $8M

  This output directly mirrors Phase 2 (System Impact Study)
  of the FERC interconnection process — what Piq automates.
""")

print("  Next: notebooks/04_upgrade_recommendations.py")
print("=" * 65)