#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze pruning effectiveness by examining SVG files
Extracts opacity distribution and stroke statistics
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_svg(svg_path):
    """Parse SVG and extract stroke opacities"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    opacities = []
    stroke_count = 0

    # Find all path and g elements
    for elem in root.iter():
        if elem.tag.endswith('path') or elem.tag.endswith('g'):
            stroke_count += 1

            # Extract stroke opacity
            style = elem.get('style', '')
            stroke_opacity = None

            # Parse style attribute
            if 'stroke-opacity' in style:
                for part in style.split(';'):
                    if 'stroke-opacity' in part:
                        stroke_opacity = float(part.split(':')[1].strip())
                        break

            # Check direct attribute
            if stroke_opacity is None:
                stroke_opacity = elem.get('stroke-opacity')
                if stroke_opacity:
                    stroke_opacity = float(stroke_opacity)

            # Check opacity attribute
            if stroke_opacity is None:
                opacity = elem.get('opacity')
                if opacity:
                    stroke_opacity = float(opacity)

            # Default to 1.0 if not found
            if stroke_opacity is None:
                stroke_opacity = 1.0

            opacities.append(stroke_opacity)

    return np.array(opacities), stroke_count


def extract_short_name(svg_path):
    """Extract a short, readable name from the path"""
    path_str = str(svg_path)

    # Try to extract test name from path structure
    # e.g., ".../baseline_128paths/..." -> "Baseline 128"
    # e.g., ".../pruned_256paths/..." -> "Pruned 256"
    if 'baseline_128' in path_str:
        return "Baseline 128"
    elif 'baseline_256' in path_str:
        return "Baseline 256"
    elif 'pruned_128' in path_str:
        return "Pruned 128"
    elif 'pruned_256' in path_str:
        return "Pruned 256"
    elif 'pruned_aggressive' in path_str or 'aggressive' in path_str:
        return "Aggressive"
    else:
        # Fallback: use parent directory name
        return Path(svg_path).parent.parent.name


def analyze_svg_file(svg_path, name=None):
    """Analyze a single SVG file"""
    if not Path(svg_path).exists():
        print(f"⚠ File not found: {svg_path}")
        return None

    # Auto-generate short name if not provided
    if name is None:
        name = extract_short_name(svg_path)

    opacities, count = parse_svg(svg_path)

    # Calculate statistics
    dead_strokes = np.sum(opacities < 0.01)
    ghost_strokes = np.sum((opacities >= 0.01) & (opacities < 0.1))
    low_strokes = np.sum((opacities >= 0.1) & (opacities < 0.3))
    visible_strokes = np.sum(opacities >= 0.3)

    stats = {
        'name': name,
        'path': svg_path,
        'total': count,
        'opacities': opacities,
        'dead': dead_strokes,
        'ghost': ghost_strokes,
        'low': low_strokes,
        'visible': visible_strokes,
        'mean_opacity': np.mean(opacities),
        'median_opacity': np.median(opacities),
        'file_size': Path(svg_path).stat().st_size
    }

    return stats


def plot_comparison(stats_list, output_path):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pruning Effectiveness Analysis', fontsize=16, fontweight='bold')

    # Subplot 1: Opacity distribution histograms
    ax = axes[0, 0]
    for stats in stats_list:
        ax.hist(stats['opacities'], bins=50, alpha=0.6, label=stats['name'])
    ax.set_xlabel('Stroke Opacity', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Opacity Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Subplot 2: Stroke category breakdown
    ax = axes[0, 1]
    categories = ['Dead\n(<0.01)', 'Ghost\n(0.01-0.1)', 'Low\n(0.1-0.3)', 'Visible\n(≥0.3)']
    x = np.arange(len(categories))
    width = 0.8 / len(stats_list)

    for i, stats in enumerate(stats_list):
        counts = [stats['dead'], stats['ghost'], stats['low'], stats['visible']]
        ax.bar(x + i * width, counts, width, label=stats['name'], alpha=0.8)

    ax.set_xlabel('Stroke Category', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Stroke Category Breakdown', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(stats_list) - 1) / 2)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Total stroke count comparison
    ax = axes[1, 0]
    names = [s['name'] for s in stats_list]
    totals = [s['total'] for s in stats_list]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    bars = ax.bar(names, totals, color=colors[:len(names)], alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Total Strokes', fontsize=11)
    ax.set_title('Final Stroke Count', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # No rotation needed for short names
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10)

    # Subplot 4: File size comparison
    ax = axes[1, 1]
    sizes_kb = [s['file_size'] / 1024 for s in stats_list]
    bars = ax.bar(names, sizes_kb, color=colors[:len(names)], alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}KB',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('File Size (KB)', fontsize=11)
    ax.set_title('SVG File Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # No rotation needed for short names
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")


def print_report(stats_list):
    """Print detailed analysis report"""
    print("\n" + "="*70)
    print("DETAILED PRUNING ANALYSIS REPORT")
    print("="*70 + "\n")

    for i, stats in enumerate(stats_list, 1):
        print(f"{i}. {stats['name']}")
        print(f"   {'─' * 60}")
        print(f"   Total Strokes:     {stats['total']}")
        print(f"   Dead (α<0.01):     {stats['dead']:4d} ({stats['dead']/stats['total']*100:5.1f}%)")
        print(f"   Ghost (0.01-0.1):  {stats['ghost']:4d} ({stats['ghost']/stats['total']*100:5.1f}%)")
        print(f"   Low (0.1-0.3):     {stats['low']:4d} ({stats['low']/stats['total']*100:5.1f}%)")
        print(f"   Visible (≥0.3):    {stats['visible']:4d} ({stats['visible']/stats['total']*100:5.1f}%)")
        print(f"   Mean Opacity:      {stats['mean_opacity']:.3f}")
        print(f"   Median Opacity:    {stats['median_opacity']:.3f}")
        print(f"   File Size:         {stats['file_size']/1024:.2f} KB")
        print()

    # Calculate improvements
    if len(stats_list) >= 2:
        baseline = stats_list[0]
        pruned = stats_list[1]

        stroke_reduction = (baseline['total'] - pruned['total']) / baseline['total'] * 100
        size_reduction = (baseline['file_size'] - pruned['file_size']) / baseline['file_size'] * 100
        dead_reduction = (baseline['dead'] - pruned['dead'])

        print("="*70)
        print("PRUNING EFFECTIVENESS METRICS")
        print("="*70)
        print(f"Stroke Reduction:    {stroke_reduction:6.1f}% ({baseline['total']} → {pruned['total']})")
        print(f"Dead Strokes Removed: {dead_reduction:6d} strokes eliminated")
        print(f"File Size Reduction:  {size_reduction:6.1f}% ({baseline['file_size']/1024:.1f}KB → {pruned['file_size']/1024:.1f}KB)")
        print(f"Opacity Improvement:  {baseline['mean_opacity']:.3f} → {pruned['mean_opacity']:.3f}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze SVG pruning effectiveness')
    parser.add_argument('svg_files', nargs='+', help='SVG files to analyze')
    parser.add_argument('-o', '--output', default='pruning_analysis.png',
                        help='Output visualization file')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    # Analyze all files
    stats_list = []
    for svg_path in args.svg_files:
        stats = analyze_svg_file(svg_path)  # Name auto-extracted in function
        if stats:
            stats_list.append(stats)

    if not stats_list:
        print("❌ No valid SVG files found!")
        return 1

    # Print report
    print_report(stats_list)

    # Generate visualization
    if not args.no_plot:
        plot_comparison(stats_list, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())