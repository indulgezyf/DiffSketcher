#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and compare benchmark results between baseline and improved versions.

Usage:
    python analyze_results.py --baseline workdir/baseline_test --improved workdir/improved_test
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple
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
            # Parse progress bar output
            # Example: "lr: 1.00, l_total: 0.1234, l_clip_fc: 0.0123, ..."
            if "l_total:" in line:
                try:
                    # Extract values
                    total_match = re.search(r"l_total:\s*([\d.]+)", line)

                    # Only process if we have total loss
                    if not total_match:
                        continue

                    clip_fc_match = re.search(r"l_clip_fc:\s*([\d.]+)", line)
                    clip_conv_match = re.search(r"l_clip_conv\(\d+\):\s*([\d.]+)", line)
                    percep_match = re.search(r"l_percep:\s*([\d.]+)", line)
                    sds_match = re.search(r"sds:\s*([\d.]+e[+-]?\d+)", line)

                    # Append values, use None for missing values to keep alignment
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

    # Find convergence step (where loss stops improving significantly)
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


def plot_comparison(baseline_losses: Dict, improved_losses: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot total loss comparison
    plt.figure(figsize=(12, 6))

    if baseline_losses and baseline_losses["total"]:
        plt.plot(
            baseline_losses["steps"],
            baseline_losses["total"],
            label="Baseline",
            color="blue",
            alpha=0.7,
            linewidth=2,
        )

    if improved_losses and improved_losses["total"]:
        plt.plot(
            improved_losses["steps"],
            improved_losses["total"],
            label="Improved (Curriculum + Dynamic Weights)",
            color="red",
            alpha=0.7,
            linewidth=2,
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.title(
        "Training Loss Comparison: Baseline vs Improved", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot_path = output_dir / "loss_comparison.png"
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Loss plot saved: {loss_plot_path}")
    plt.close()

    # Plot individual loss components
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_types = [
        ("clip_fc", "CLIP FC Loss"),
        ("clip_conv", "CLIP Conv Loss"),
        ("percep", "Perceptual Loss"),
        ("sds", "SDS Gradient"),
    ]

    for idx, (loss_key, loss_name) in enumerate(loss_types):
        ax = axes[idx // 2, idx % 2]

        if baseline_losses and baseline_losses.get(loss_key):
            # Filter out None values
            steps = []
            values = []
            for s, v in zip(baseline_losses["steps"], baseline_losses[loss_key]):
                if v is not None:
                    steps.append(s)
                    values.append(v)

            if steps:
                ax.plot(
                    steps,
                    values,
                    label="Baseline",
                    color="blue",
                    alpha=0.7,
                )

        if improved_losses and improved_losses.get(loss_key):
            # Filter out None values
            steps = []
            values = []
            for s, v in zip(improved_losses["steps"], improved_losses[loss_key]):
                if v is not None:
                    steps.append(s)
                    values.append(v)

            if steps:
                ax.plot(
                    steps,
                    values,
                    label="Improved",
                    color="red",
                    alpha=0.7,
                )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(loss_name)
        ax.set_title(loss_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    components_plot_path = output_dir / "loss_components.png"
    plt.savefig(components_plot_path, dpi=150)
    print(f"  📊 Components plot saved: {components_plot_path}")
    plt.close()


def print_comparison(baseline_metrics: Dict, improved_metrics: Dict):
    """Print comparison results."""
    print("\n" + "=" * 80)
    print("📊 METRICS COMPARISON")
    print("=" * 80 + "\n")

    if not baseline_metrics or not improved_metrics:
        print("❌ Insufficient data for comparison")
        return

    print(f"{'Metric':<30} {'Baseline':>15} {'Improved':>15} {'Change':>15}")
    print("-" * 80)

    # Final loss
    baseline_final = baseline_metrics["final_total_loss"]
    improved_final = improved_metrics["final_total_loss"]
    if baseline_final and improved_final:
        change = (baseline_final - improved_final) / baseline_final * 100
        print(
            f"{'Final Total Loss':<30} {baseline_final:>15.4f} {improved_final:>15.4f} {change:>14.1f}%"
        )

    # Min loss
    baseline_min = baseline_metrics["min_total_loss"]
    improved_min = improved_metrics["min_total_loss"]
    if baseline_min and improved_min:
        change = (baseline_min - improved_min) / baseline_min * 100
        print(
            f"{'Min Total Loss':<30} {baseline_min:>15.4f} {improved_min:>15.4f} {change:>14.1f}%"
        )

    # Convergence speed
    baseline_conv = baseline_metrics["convergence_step"]
    improved_conv = improved_metrics["convergence_step"]
    if baseline_conv and improved_conv:
        change = (baseline_conv - improved_conv) / baseline_conv * 100
        print(
            f"{'Convergence Step (90%)':<30} {baseline_conv:>15d} {improved_conv:>15d} {change:>14.1f}%"
        )

    # Loss reduction
    baseline_reduction = baseline_metrics["loss_reduction"]
    improved_reduction = improved_metrics["loss_reduction"]
    if baseline_reduction and improved_reduction:
        print(
            f"{'Loss Reduction':<30} {baseline_reduction:>14.1f}% {improved_reduction:>14.1f}%"
        )

    print("\n" + "=" * 80 + "\n")

    # Summary
    if baseline_final and improved_final and baseline_conv and improved_conv:
        quality_improvement = (baseline_final - improved_final) / baseline_final * 100
        speed_improvement = (baseline_conv - improved_conv) / baseline_conv * 100

        print("SUMMARY:")
        print(
            f"  Quality Improvement: {quality_improvement:+.1f}% (lower loss is better)"
        )
        print(
            f"  Convergence Speed:   {speed_improvement:+.1f}% (negative means faster)"
        )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DiffSketcher benchmark results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="./workdir/baseline_test",
        help="Path to baseline results directory",
    )
    parser.add_argument(
        "--improved",
        type=str,
        default="./workdir/improved_test",
        help="Path to improved results directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./workdir/analysis",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    improved_dir = Path(args.improved)
    output_dir = Path(args.output)

    print("\n" + "=" * 80)
    print("🔍 Analyzing DiffSketcher Results")
    print("=" * 80)
    print(f"\nBaseline: {baseline_dir}")
    print(f"Improved: {improved_dir}")
    print(f"Output:   {output_dir}\n")

    # Parse logs
    print("📖 Parsing training logs...")
    baseline_losses = parse_training_log(baseline_dir / "training.log")
    improved_losses = parse_training_log(improved_dir / "training.log")

    if not baseline_losses:
        print(f"⚠️  Warning: Could not parse baseline log")
    else:
        print(f"  ✓ Baseline: {len(baseline_losses['total'])} steps parsed")

    if not improved_losses:
        print(f"⚠️  Warning: Could not parse improved log")
    else:
        print(f"  ✓ Improved: {len(improved_losses['total'])} steps parsed")

    # Calculate metrics
    print("\n📈 Calculating metrics...")
    baseline_metrics = calculate_metrics(baseline_losses) if baseline_losses else None
    improved_metrics = calculate_metrics(improved_losses) if improved_losses else None

    # Generate plots
    if baseline_losses or improved_losses:
        print("\n🎨 Generating plots...")
        plot_comparison(baseline_losses, improved_losses, output_dir)

    # Print comparison
    if baseline_metrics and improved_metrics:
        print_comparison(baseline_metrics, improved_metrics)
    else:
        print("\n❌ Insufficient data for comparison")

    print("✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
