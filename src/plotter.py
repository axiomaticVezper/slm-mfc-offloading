import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["local_only", "greedy", "static_threshold", "always_fog",
         "slm_noguard", "slm_guard", "slm_noguard_faulty", "slm_guard_faulty", "slm_guard_charged"]
LABELS = {
    "local_only": "Local-only", "greedy": "Greedy", "static_threshold": "Static thr.",
    "always_fog": "Always-Fog", "slm_noguard": "SLM (no guard)", "slm_guard": "SLM + guard",
    "slm_noguard_faulty": "SLM no guard\n(20% faults)", "slm_guard_faulty": "SLM + guard\n(20% faults)",
    "slm_guard_charged": "SLM + guard\n(+inference time)",
}


def _stats(runs, scenario, metric, configs):
    sub = runs[runs.scenario == scenario]
    means, stds = [], []
    for c in configs:
        v = sub[sub.config == c][metric]
        means.append(v.mean() if len(v) else 0.0)
        stds.append(v.std() if len(v) > 1 else 0.0)
    return np.nan_to_num(means), np.nan_to_num(stds)


def _bars(runs, metric, ylabel, title, path, configs=None):
    configs = [c for c in (configs or ORDER) if c in set(runs.config)]
    scenarios = list(runs.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        m, s = _stats(runs, sc, metric, configs)
        ax.bar(range(len(configs)), m, yerr=s, capsize=3)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([LABELS[c] for c in configs], rotation=40, ha="right", fontsize=8)
        ax.set_title(sc)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_evaluation_plots(runs, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    f = lambda n: os.path.join(figures_dir, n)
    _bars(runs, "deadline_hit_rate_pct", "Deadline hit rate (%)", "Deadline satisfaction (mean ± std over seeds)", f("deadline_satisfaction.png"))
    _bars(runs, "avg_energy_j", "Energy per executed task (J)", "Energy consumption (mean ± std over seeds)", f("energy_consumption.png"))
    _bars(runs, "avg_turnaround_s", "Turnaround (s)", "Average turnaround of executed tasks (mean ± std)", f("turnaround_latency.png"))

    # Tier distribution (stacked)
    configs = [c for c in ORDER if c in set(runs.config)]
    scenarios = list(runs.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        bottom = np.zeros(len(configs))
        for col, name in [("mist_pct", "Mist"), ("fog_pct", "Fog"), ("cloud_pct", "Cloud"), ("rejected_pct", "Rejected")]:
            m, _ = _stats(runs, sc, col, configs)
            ax.bar(range(len(configs)), m, bottom=bottom, label=name)
            bottom += m
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([LABELS[c] for c in configs], rotation=40, ha="right", fontsize=8)
        ax.set_title(sc)
    np.atleast_1d(axes)[0].set_ylabel("Share of tasks (%)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Where tasks were executed")
    fig.tight_layout()
    fig.savefig(f("tier_distribution.png"), dpi=200)
    plt.close(fig)

    # Guard ablation
    abl = [c for c in ["slm_noguard", "slm_guard", "slm_noguard_faulty", "slm_guard_faulty"] if c in set(runs.config)]
    metrics = [("deadline_hit_rate_pct", "Deadline hit (%)"), ("mem_violation_pct", "Memory violations (%)"),
               ("overloaded_pct", "Overloaded placements (%)")]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5), sharey=True)
    w = 0.8 / max(len(abl), 1)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        for i, c in enumerate(abl):
            vals = [_stats(runs, sc, m, [c])[0][0] for m, _ in metrics]
            errs = [_stats(runs, sc, m, [c])[1][0] for m, _ in metrics]
            ax.bar(np.arange(len(metrics)) + i * w, vals, w, yerr=errs, capsize=2, label=LABELS[c].replace("\n", " "))
        ax.set_xticks(np.arange(len(metrics)) + w * (len(abl) - 1) / 2)
        ax.set_xticklabels([n for _, n in metrics], fontsize=9)
        ax.set_title(sc)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Effect of the resource-feasibility guard")
    fig.tight_layout()
    fig.savefig(f("guard_ablation.png"), dpi=200)
    plt.close(fig)