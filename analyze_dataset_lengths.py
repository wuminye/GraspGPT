#!/usr/bin/env python3
"""
Analyze length distribution of samples in VoxelDataset
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import time

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import required modules
from graspGPT.model.dataset import VoxelDataset
from graspGPT.model.precomputed_dataset import PrecomputedDataset

def analyze_sample_lengths(dataset, sample_size=None):
    """Analyze sample length distribution"""
    print(f"Starting sample length distribution analysis...")
    
    # If sample_size is specified, analyze only a subset of samples
    total_samples = len(dataset)
    if sample_size and sample_size < total_samples:
        indices = np.random.choice(total_samples, sample_size, replace=False)
        print(f"Randomly sampling {sample_size} samples for analysis (total {total_samples} samples)")
    else:
        indices = range(total_samples)
        print(f"Analyzing all {total_samples} samples")
    
    lengths = []
    voxel_counts = []
    color_group_counts = []
    
    start_time = time.time()
    
    for i, idx in enumerate(indices):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(indices) - i) / rate
            print(f"Progress: {i}/{len(indices)} ({i/len(indices)*100:.1f}%) - "
                  f"Speed: {rate:.1f} samples/s - Estimated remaining: {remaining:.1f}s")
        
        try:
            # Get data through __getitem__
            # PrecomputedDataset returns: (tokens, max_sequence_length, scene_grasps)
            tokens, max_seq_len, scene_grasps = dataset[idx]
            
            # Get token sequence length
            if isinstance(tokens, torch.Tensor):
                token_length = tokens.shape[0]  # sequence length
            else:
                token_length = len(tokens)
            
            # Get voxel and color group information from raw data
            sample = dataset.data[idx]
            voxel_data = sample['voxel_data']
            
            # Calculate total voxels and color groups count
            total_voxels = 0
            color_groups = set()
            
            for color, coordinates in voxel_data:
                if hasattr(coordinates, 'shape'):
                    total_voxels += coordinates.shape[0]
                else:
                    total_voxels += len(coordinates)
                color_groups.add(color)
            
            num_color_groups = len(color_groups)
            
            lengths.append(token_length)
            voxel_counts.append(total_voxels)
            color_group_counts.append(num_color_groups)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed, time elapsed: {elapsed_time:.2f}s")
    
    return np.array(lengths), np.array(voxel_counts), np.array(color_group_counts)

def print_statistics(data, name):
    """Print statistics"""
    print(f"\n=== {name} Statistics ===")
    print(f"Sample count: {len(data)}")
    print(f"Minimum: {np.min(data)}")
    print(f"Maximum: {np.max(data)}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard deviation: {np.std(data):.2f}")
    print(f"25th percentile: {np.percentile(data, 25):.2f}")
    print(f"75th percentile: {np.percentile(data, 75):.2f}")
    print(f"95th percentile: {np.percentile(data, 95):.2f}")
    print(f"99th percentile: {np.percentile(data, 99):.2f}")

def plot_distribution(data, name, bins=50):
    """Plot distribution histogram"""
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(f'{name}')
    plt.ylabel('Frequency')
    plt.title(f'{name} Distribution Histogram')
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution function
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y)
    plt.xlabel(f'{name}')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{name} Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{name.lower().replace(" ", "_")}_distribution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved: {filename}")
    plt.close()

def analyze_length_vs_voxels(lengths, voxel_counts):
    """Analyze relationship between token length and voxel count"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(voxel_counts, lengths, alpha=0.5, s=1)
    plt.xlabel('Voxel Count')
    plt.ylabel('Token Sequence Length')
    plt.title('Token Length vs Voxel Count')
    plt.grid(True, alpha=0.3)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(voxel_counts, lengths)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Ratio distribution
    plt.subplot(1, 2, 2)
    ratios = lengths / voxel_counts
    valid_ratios = ratios[np.isfinite(ratios)]  # exclude infinite values
    plt.hist(valid_ratios, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Token Length/Voxel Count Ratio')
    plt.ylabel('Frequency')
    plt.title('Token Length to Voxel Count Ratio Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('length_vs_voxels_analysis.png', dpi=150, bbox_inches='tight')
    print("Token length vs voxel analysis plot saved: length_vs_voxels_analysis.png")
    plt.close()
    
    print(f"\n=== Token Length vs Voxel Count Analysis ===")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Average ratio: {np.mean(valid_ratios):.4f}")
    print(f"Ratio standard deviation: {np.std(valid_ratios):.4f}")

def main():
    # Set data path
    data_path = "output/precomputed_data/"
    
    print("=== Creating VoxelDataset ===")
    try:
        dataset = PrecomputedDataset(
            data_path=data_path,
            max_sequence_length=8192
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Vocabulary size: {dataset.get_vocab_size()}")
        print(f"Volume dimensions: {dataset.volume_dims}")
        
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return
    
    # Analyze sample length distribution
    # For large datasets, can set sample_size for sampling analysis
    sample_size = 5000 if len(dataset) > 5000 else None
    
    lengths, voxel_counts, color_group_counts = analyze_sample_lengths(dataset, sample_size)
    
    if len(lengths) == 0:
        print("No successfully processed samples")
        return
    
    # Print statistics
    print_statistics(lengths, "Token Sequence Length")
    print_statistics(voxel_counts, "Voxel Count")
    print_statistics(color_group_counts, "Color Group Count")
    
    # Plot distributions
    plot_distribution(lengths, "Token Sequence Length")
    plot_distribution(voxel_counts, "Voxel Count")
    plot_distribution(color_group_counts, "Color Group Count")
    
    # Analyze relationship between length and voxel count
    analyze_length_vs_voxels(lengths, voxel_counts)
    
    # Find longest and shortest samples
    print(f"\n=== Extreme Sample Analysis ===")
    if sample_size:
        print("Note: Following analysis is based on sampled data")
    
    max_idx = np.argmax(lengths)
    min_idx = np.argmin(lengths)
    
    print(f"Longest sample: length={lengths[max_idx]}, voxels={voxel_counts[max_idx]}, color_groups={color_group_counts[max_idx]}")
    print(f"Shortest sample: length={lengths[min_idx]}, voxels={voxel_counts[min_idx]}, color_groups={color_group_counts[min_idx]}")
    
    # Count samples exceeding maximum sequence length
    max_seq_len = dataset.max_sequence_length
    over_limit = np.sum(lengths > max_seq_len)
    print(f"\nSamples exceeding max sequence length ({max_seq_len}): {over_limit} ({over_limit/len(lengths)*100:.2f}%)")
    
    # Distribution across different length intervals
    print(f"\n=== Length Interval Distribution ===")
    bins = [0, 100, 500, 1000, 1500, 2000, 3000, 5000,  max_seq_len, np.inf]
    labels = ['<100', '100-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000-5000', f'5000-{max_seq_len}', f'>{max_seq_len}']
    
    hist, _ = np.histogram(lengths, bins=bins)
    for i, (label, count) in enumerate(zip(labels, hist)):
        percentage = count / len(lengths) * 100
        print(f"{label:>12}: {count:>6} samples ({percentage:>5.1f}%)")

if __name__ == "__main__":
    main()