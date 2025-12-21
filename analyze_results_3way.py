#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and compare three-way benchmark results.

Usage:
    python analyze_results_3way.py
"""

import argparse
import re
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_file: Path) -> Dict:
    """Parse training log and extract loss metrics over time."""
    if not log_file.exists():
        return None

    losses = {
        "total": [],
        "clip_fc": [],
        "clip_conv": [],
        "percep": [],
        "sds": [],
        "steps": [],
    }

    with open(log_file, "r") as f:
        for line in f:
            if "l_total:" in line:
                try:
                    total_match = re.search(r"l_total:\s*([\d.]+)", line)

                    if not total_match:
                        continue

                    clip_fc_match = re.search(r"l_clip_fc:\s*([\d.]+)", line)
                    clip_conv_match = re.search(r"l_clip_conv\(\d+\):\s*([\d.]+)", line)
                    percep_match = re.search(r"l_percep:\s*([\d.]+)", line)
                    sds_match = re.search(r"sds:\s*([\d.]+e[+-]?\d+)", line)

                    losses["total"].append(float(total_match.group(1)))
                    losses["clip_fc"].append(float(clip_fc_match.group(1)) if clip_fc_match else None)
                    losses["clip_conv"].append(float(clip_conv_match.group(1)) if clip_conv_match else None)
                    losses["percep"].append(float(percep_match.group(1)) if percep_match else None)
                    losses["sds"].append(float(sds_match.group(1)) if sds_match else None)
                    losses["steps"].append(len(losses["total"]))

                except Exception:
                    continue

    return losses if losses["total"] else None


def calculate_metrics(losses: Dict) -> Dict:
    """Calculate summary metrics from losses."""
    if not losses or not losses["total"]:
        return None

    metrics = {
        "final_total_loss": losses["total"][-1] if losses["total"] else None,
        "min_total_loss": min(losses["total"]) if losses["total"] else None,
        "avg_total_loss": np.mean(losses["total"]) if losses["total"] else None,
        "convergence_step": None,
        "loss_reduction": None,
    }

    if len(losses["total"]) > 10:
        initial_loss = np.mean(losses["total"][:10])
        final_loss = metrics["final_total_loss"]
        metrics["loss_reduction"] = (initial_loss - final_loss) / initial_loss * 100

        # Find step where 90% of improvement is achieved
        target_loss = initial_loss - 0.9 * (initial_loss - final_loss)
        for i, loss in enumerate(losses["total"]):
            if loss <= target_loss:
                metrics["convergence_step"] = losses["steps"][i]
                break

    return metrics


def plot_3way_comparison(baseline_losses: Dict, dynamic_weights_losses: Dict,
                         full_improved_losses: Dict, output_dir: Path):
    """Generate three-way comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot total loss comparison
    plt.figure(figsize=(14, 7))

    if baseline_losses and baseline_losses["total"]:
        plt.plot(
            baseline_losses["steps"],
            baseline_losses["total"],
            label="Baseline (No Improvements)",
            color="blue",
            alpha=0.8,
            linewidth=2,
        )

    if dynamic_weights_losses and dynamic_weights_losses["total"]:
        plt.plot(
            dynamic_weights_losses["steps"],
            dynamic_weights_losses["total"],
            label="Dynamic Weights Only",
            color="orange",
            alpha=0.8,
            linewidth=2,
        )

    if full_improved_losses and full_improved_losses["total"]:
        plt.plot(
            full_improved_losses["steps"],
            full_improved_losses["total"],
            label="Full Improved (Dynamic Weights + Curriculum)",
            color="red",
            alpha=0.8,
            linewidth=2,
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.title("Training Loss Comparison: Three-Way Benchmark", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot_path = output_dir / "loss_comparison_3way.png"
    plt.savefig(loss_plot_path, dpi=150)
    print(f"📊 Loss comparison plot saved: {loss_plot_path}")
    plt.close()

    # Plot individual loss components (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    loss_types = [
        ("clip_fc", "CLIP FC Loss"),
        ("clip_conv", "CLIP Conv Loss"),
        ("percep", "Perceptual Loss"),
        ("sds", "SDS Gradient"),
    ]

    colors = {"baseline": "blue", "dynamic": "orange", "full": "red"}
    labels = {
        "baseline": "Baseline",
        "dynamic": "Dynamic Weights",
        "full": "Full Improved"
    }

    for idx, (loss_key, loss_name) in enumerate(loss_types):
        ax = axes[idx // 2, idx % 2]

        # Plot baseline
        if baseline_losses and baseline_losses.get(loss_key):
            steps, values = [], []
            for s, v in zip(baseline_losses["steps"], baseline_losses[loss_key]):
                if v is not None:
                    steps.append(s)
                    values.append(v)
            if steps:
                ax.plot(steps, values, label=labels["baseline"],
                       color=colors["baseline"], alpha=0.7, linewidth=1.5)

        # Plot dynamic weights
        if dynamic_weights_losses and dynamic_weights_losses.get(loss_key):
            steps, values = [], []
            for s, v in zip(dynamic_weights_losses["steps"], dynamic_weights_losses[loss_key]):
                if v is not None:
                    steps.append(s)
                    values.append(v)
            if steps:
                ax.plot(steps, values, label=labels["dynamic"],
                       color=colors["dynamic"], alpha=0.7, linewidth=1.5)

        # Plot full improved
        if full_improved_losses and full_improved_losses.get(loss_key):
            steps, values = [], []
            for s, v in zip(full_improved_losses["steps"], full_improved_losses[loss_key]):
                if v is not None:
                    steps.append(s)
                    values.append(v)
            if steps:
                ax.plot(steps, values, label=labels["full"],
                       color=colors["full"], alpha=0.7, linewidth=1.5)

        ax.set_xlabel("Training Step")
        ax.set_ylabel(loss_name)
        ax.set_title(loss_name, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    components_plot_path = output_dir / "loss_components_3way.png"
    plt.savefig(components_plot_path, dpi=150)
    print(f"📊 Components plot saved: {components_plot_path}")
    plt.close()


def print_3way_comparison(baseline_metrics: Dict, dynamic_weights_metrics: Dict,
                          full_improved_metrics: Dict):
    """Print three-way comparison results."""
    print("\n" + "=" * 100)
    print("📊 THREE-WAY METRICS COMPARISON")
    print("=" * 100 + "\n")

    if not baseline_metrics or not dynamic_weights_metrics or not full_improved_metrics:
        print("❌ Insufficient data for comparison")
        return

    print(f"{'Metric':<30} {'Baseline':>18} {'Dynamic Weights':>18} {'Full Improved':>18}")
    print("-" * 100)

    # Final loss
    baseline_final = baseline_metrics["final_total_loss"]
    dynamic_final = dynamic_weights_metrics["final_total_loss"]
    full_final = full_improved_metrics["final_total_loss"]

    if baseline_final and dynamic_final and full_final:
        print(f"{'Final Total Loss':<30} {baseline_final:>18.4f} {dynamic_final:>18.4f} {full_final:>18.4f}")

    # Min loss
    baseline_min = baseline_metrics["min_total_loss"]
    dynamic_min = dynamic_weights_metrics["min_total_loss"]
    full_min = full_improved_metrics["min_total_loss"]

    if baseline_min and dynamic_min and full_min:
        print(f"{'Min Total Loss':<30} {baseline_min:>18.4f} {dynamic_min:>18.4f} {full_min:>18.4f}")

    # Convergence speed
    baseline_conv = baseline_metrics["convergence_step"]
    dynamic_conv = dynamic_weights_metrics["convergence_step"]
    full_conv = full_improved_metrics["convergence_step"]

    if baseline_conv and dynamic_conv and full_conv:
        print(f"{'Convergence Step (90%)':<30} {baseline_conv:>18d} {dynamic_conv:>18d} {full_conv:>18d}")

    # Loss reduction
    baseline_reduction = baseline_metrics["loss_reduction"]
    dynamic_reduction = dynamic_weights_metrics["loss_reduction"]
    full_reduction = full_improved_metrics["loss_reduction"]

    if baseline_reduction and dynamic_reduction and full_reduction:
        print(f"{'Loss Reduction':<30} {baseline_reduction:>17.1f}% {dynamic_reduction:>17.1f}% {full_reduction:>17.1f}%")

    print("\n" + "=" * 100 + "\n")

    # Summary
    if baseline_final and dynamic_final and full_final:
        print("SUMMARY:")
        print(f"  Baseline → Dynamic Weights: {((baseline_final - dynamic_final) / baseline_final * 100):+.1f}% improvement")
        print(f"  Baseline → Full Improved:   {((baseline_final - full_final) / baseline_final * 100):+.1f}% improvement")
        print(f"  Dynamic → Full Improved:    {((dynamic_final - full_final) / dynamic_final * 100):+.1f}% improvement")
        print()

        if baseline_conv and dynamic_conv and full_conv:
            print("CONVERGENCE SPEED:")
            print(f"  Baseline → Dynamic Weights: {((baseline_conv - dynamic_conv) / baseline_conv * 100):+.1f}% faster")
            print(f"  Baseline → Full Improved:   {((baseline_conv - full_conv) / baseline_conv * 100):+.1f}% faster")
            print(f"  Dynamic → Full Improved:    {((dynamic_conv - full_conv) / dynamic_conv * 100):+.1f}% faster")
            print()


def main():
    parser = argparse.ArgumentParser(description="Analyze DiffSketcher three-way benchmark results")
    parser.add_argument("--baseline", type=str, default="./workdir/baseline_test",
                       help="Path to baseline results directory")
    parser.add_argument("--dynamic", type=str, default="./workdir/dynamic_weights_test",
                       help="Path to dynamic weights results directory")
    parser.add_argument("--full", type=str, default="./workdir/full_improved_test",
                       help="Path to full improved results directory")
    parser.add_argument("--output", type=str, default="./workdir/analysis_3way",
                       help="Output directory for plots")

    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    dynamic_dir = Path(args.dynamic)
    full_dir = Path(args.full)
    output_dir = Path(args.output)

    print("\n" + "=" * 100)
    print("🔍 Analyzing DiffSketcher Three-Way Results")
    print("=" * 100)
    print(f"\nBaseline:        {baseline_dir}")
    print(f"Dynamic Weights: {dynamic_dir}")
    print(f"Full Improved:   {full_dir}")
    print(f"Output:          {output_dir}\n")

    # Parse logs
    print("📖 Parsing training logs...")
    baseline_losses = parse_training_log(baseline_dir / "training.log")
    dynamic_losses = parse_training_log(dynamic_dir / "training.log")
    full_losses = parse_training_log(full_dir / "training.log")

    if not baseline_losses:
        print(f"⚠️  Warning: Could not parse baseline log")
    else:
        print(f"  ✓ Baseline: {len(baseline_losses['total'])} steps parsed")

    if not dynamic_losses:
        print(f"⚠️  Warning: Could not parse dynamic weights log")
    else:
        print(f"  ✓ Dynamic Weights: {len(dynamic_losses['total'])} steps parsed")

    if not full_losses:
        print(f"⚠️  Warning: Could not parse full improved log")
    else:
        print(f"  ✓ Full Improved: {len(full_losses['total'])} steps parsed")

    # Calculate metrics
    print("\n📈 Calculating metrics...")
    baseline_metrics = calculate_metrics(baseline_losses) if baseline_losses else None
    dynamic_metrics = calculate_metrics(dynamic_losses) if dynamic_losses else None
    full_metrics = calculate_metrics(full_losses) if full_losses else None

    # Generate plots
    if baseline_losses or dynamic_losses or full_losses:
        print("\n🎨 Generating plots...")
        plot_3way_comparison(baseline_losses, dynamic_losses, full_losses, output_dir)

    # Print comparison
    if baseline_metrics and dynamic_metrics and full_metrics:
        print_3way_comparison(baseline_metrics, dynamic_metrics, full_metrics)
    else:
        print("\n❌ Insufficient data for comparison")

    print("✅ Analysis complete!\n")


if __name__ == "__main__":
    main()