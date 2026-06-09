# ============================================================
# WattWorker EV Charging Grid Analysis
# Notebook 01: Base Case Power Flow
# ------------------------------------------------------------
# Two networks in parallel:
#   - IEEE 39-bus  (New England — rapid iteration)
#   - IEEE 118-bus (Production scale — interview portfolio)
#
# Scenario: EV charging hubs at two locations:
#   - Strong grid (near generation, well-connected substation)
#   - Weak grid   (end of long feeder, limited reactive support)
# ============================================================

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────
# Bus numbers chosen to represent Strong vs Weak grid locations
# 39-bus:  Bus 16 (near Gen) vs Bus 27 (end of feeder)
# 118-bus: Bus 49 (near Gen) vs Bus 106 (end of feeder)

NETWORKS = {
    "IEEE 39-bus": {
        "loader"     : pn.case39,
        "bus_strong" : 16,
        "bus_weak"   : 27,
        "label_strong": "Bay Area Port (strong grid)",
        "label_weak"  : "Suburban Feeder (weak grid)",
    },
    "IEEE 118-bus": {
        "loader"     : pn.case118,
        "bus_strong" : 49,
        "bus_weak"   : 106,
        "label_strong": "Bay Area Port (strong grid)",
        "label_weak"  : "Suburban Feeder (weak grid)",
    },
}

# ── ANALYSIS FUNCTION ─────────────────────────────────────────
def run_base_case(name, config):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)

    # Load network
    net = config["loader"]()
    print(f"  Buses      : {len(net.bus)}")
    print(f"  Lines      : {len(net.line)}")
    print(f"  Generators : {len(net.gen)}")
    print(f"  Loads      : {len(net.load)}")

    # Run AC power flow (Newton-Raphson)
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
    print(f"\n  Power flow : CONVERGED ✓")

    # ── Voltage summary ───────────────────────────────────────
    v = net.res_bus['vm_pu']
    v_violations = ((v < 0.95) | (v > 1.05)).sum()
    print(f"\n  VOLTAGE:")
    print(f"    Min : {v.min():.4f} pu  (Bus {v.idxmin()})")
    print(f"    Max : {v.max():.4f} pu  (Bus {v.idxmax()})")
    print(f"    Violations (outside 0.95-1.05 pu) : {v_violations}")

    # ── Thermal summary ───────────────────────────────────────
    loading = net.res_line['loading_percent']
    overloaded = (loading > 100).sum()
    heavy = (loading > 80).sum()
    print(f"\n  THERMAL:")
    print(f"    Max loading : {loading.max():.1f}%  (Line {loading.idxmax()})")
    print(f"    Overloaded (>100%) : {overloaded} lines")
    print(f"    Heavy load (>80%)  : {heavy} lines")

    # ── EV charging bus assessment ────────────────────────────
    bs = config["bus_strong"]
    bw = config["bus_weak"]
    print(f"\n  EV CHARGING LOCATIONS:")
    print(f"    Bus {bs} — {config['label_strong']}")
    print(f"      Voltage : {net.res_bus.loc[bs, 'vm_pu']:.4f} pu")
    print(f"    Bus {bw} — {config['label_weak']}")
    print(f"      Voltage : {net.res_bus.loc[bw, 'vm_pu']:.4f} pu")

    # ── Reactive power headroom ───────────────────────────────
    if len(net.res_gen) > 0 and 'q_mvar' in net.res_gen.columns:
        gen_q = net.res_gen['q_mvar']
        print(f"\n  REACTIVE POWER (Q):")
        print(f"    Total Q generated : {gen_q.sum():.1f} MVAR")
        print(f"    Generators absorbing Q : {(gen_q < 0).sum()}")

    print(f"\n  STATUS: {'✓ HEALTHY — no violations' if v_violations == 0 and overloaded == 0 else '⚠ HAS VIOLATIONS'}")

    return net, v, loading

# ── RUN BOTH NETWORKS ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  WattWorker EV Hub — Base Case Power Flow Analysis")
print("  Using IEEE Standard Test Cases as grid proxy")
print("=" * 60)

results = {}
for name, config in NETWORKS.items():
    net, v, loading = run_base_case(name, config)
    results[name] = {"net": net, "v": v, "loading": loading, "config": config}

# ── SIDE-BY-SIDE VISUALISATION ────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle("WattWorker EV Charging Grid Analysis — Base Case\n"
             "IEEE 39-bus vs IEEE 118-bus", fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

for col, (name, data) in enumerate(results.items()):
    v       = data["v"]
    loading = data["loading"]
    config  = data["config"]
    bs      = config["bus_strong"]
    bw      = config["bus_weak"]

    # ── Voltage plot ──────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, col])
    colors_v = ['#d73027' if vv < 0.95 or vv > 1.05 else
                '#fc8d59' if vv < 0.97 or vv > 1.03 else
                '#4dac26'
                for vv in v]
    ax_v.bar(v.index, v.values, color=colors_v, width=0.6, alpha=0.85)
    ax_v.axhline(0.95, color='red',  linestyle='--', lw=1.2, label='Limits (0.95/1.05 pu)')
    ax_v.axhline(1.05, color='red',  linestyle='--', lw=1.2)
    ax_v.axhline(1.00, color='gray', linestyle=':',  lw=0.8)

    # Mark EV buses
    for bus, marker_label in [(bs, "Strong grid"), (bw, "Weak grid")]:
        ax_v.scatter(bus, v.loc[bus],
                     s=120, zorder=6,
                     color='blue' if bus == bs else 'purple',
                     label=f"Bus {bus} ({marker_label})")

    ax_v.set_xlabel('Bus', fontsize=9)
    ax_v.set_ylabel('Voltage (pu)', fontsize=9)
    ax_v.set_title(f'{name}\nBus Voltages — Base Case', fontsize=10, fontweight='bold')
    ax_v.legend(fontsize=7, loc='lower right')
    ax_v.set_ylim(0.88, 1.12)

    # Violation annotation
    n_viol = ((v < 0.95) | (v > 1.05)).sum()
    ax_v.text(0.02, 0.97,
              f"Violations: {n_viol}",
              transform=ax_v.transAxes,
              fontsize=9, color='red' if n_viol > 0 else 'green',
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Thermal loading plot ──────────────────────────────────
    ax_l = fig.add_subplot(gs[1, col])
    colors_l = ['#d73027' if ll > 100 else
                '#fc8d59' if ll > 80  else
                '#4575b4'
                for ll in loading]
    ax_l.bar(loading.index, loading.values, color=colors_l, width=0.6, alpha=0.85)
    ax_l.axhline(100, color='red',    linestyle='--', lw=1.2, label='Thermal limit (100%)')
    ax_l.axhline(80,  color='orange', linestyle='--', lw=1.2, label='Warning (80%)')

    ax_l.set_xlabel('Line', fontsize=9)
    ax_l.set_ylabel('Loading (%)', fontsize=9)
    ax_l.set_title(f'{name}\nLine Thermal Loading — Base Case', fontsize=10, fontweight='bold')
    ax_l.legend(fontsize=7)

    n_over = (loading > 100).sum()
    ax_l.text(0.02, 0.97,
              f"Overloaded: {n_over}",
              transform=ax_l.transAxes,
              fontsize=9, color='red' if n_over > 0 else 'green',
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.savefig('outputs/01_base_case.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Plot saved → outputs/01_base_case.png")

# ── COMPARISON TABLE ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  NETWORK COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Metric':<35} {'39-bus':>10} {'118-bus':>10}")
print(f"  {'-'*55}")
r39  = results["IEEE 39-bus"]
r118 = results["IEEE 118-bus"]
rows = [
    ("Buses",                   len(r39['net'].bus),        len(r118['net'].bus)),
    ("Lines",                   len(r39['net'].line),       len(r118['net'].line)),
    ("Generators",              len(r39['net'].gen),        len(r118['net'].gen)),
    ("Min voltage (pu)",        f"{r39['v'].min():.4f}",    f"{r118['v'].min():.4f}"),
    ("Max voltage (pu)",        f"{r39['v'].max():.4f}",    f"{r118['v'].max():.4f}"),
    ("Voltage violations",      int(((r39['v']<0.95)|(r39['v']>1.05)).sum()),
                                int(((r118['v']<0.95)|(r118['v']>1.05)).sum())),
    ("Max line loading (%)",    f"{r39['loading'].max():.1f}",
                                f"{r118['loading'].max():.1f}"),
    ("Overloaded lines",        int((r39['loading']>100).sum()),
                                int((r118['loading']>100).sum())),
]
for label, v39, v118 in rows:
    print(f"  {label:<35} {str(v39):>10} {str(v118):>10}")

print("\n" + "=" * 60)
print("  BASE CASE COMPLETE")
print("  Next: notebooks/02_ev_load_injection.py")
print("  → Add EV charging load and see what breaks")
print("=" * 60)