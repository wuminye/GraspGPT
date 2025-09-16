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
from graspGPT.model.parser_and_serializer import Parser, GRASP, SB

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
            
            # Get voxel and color group information from decoded tokens
            sample = dataset.data[idx]
            raw_tokens = sample['raw_tokens']
            
            # Decode tokens to get original token list
            from graspGPT.model.token_manager import decode_sequence
            decoded_tokens = decode_sequence(raw_tokens, dataset.token_mapping)
            
            # Parse tokens to AST to extract voxel information
            try:
                parser = Parser(decoded_tokens)
                ast = parser.parse()
                
                # Calculate total voxels and color groups (object tags) count
                total_voxels = 0
                color_groups = set()
                
                # Count voxels and object tags from SB items
                for item in ast.items:
                    if isinstance(item, SB):
                        # Count coordinate blocks (CBs) as voxels
                        total_voxels += len(item.cbs)
                        # Add object tag as color group equivalent
                        color_groups.add(item.tag)
                
                num_color_groups = len(color_groups)
                
            except Exception as parse_error:
                print(f"Warning: Failed to parse tokens for sample {idx}: {parse_error}")
                # Fallback: estimate from token count
                total_voxels = len(raw_tokens) // 10  # rough estimate
                num_color_groups = 1  # minimal estimate
            
            lengths.append(token_length)
            voxel_counts.append(total_voxels)
            color_group_counts.append(num_color_groups)
            
        except Exception as e:
            print(f"Error processing sample shape {idx}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed, time elapsed: {elapsed_time:.2f}s")
    
    return np.array(lengths), np.array(voxel_counts), np.array(color_group_counts)


def analyze_grasp_distribution(dataset, sample_size=None):
    """Analyze grasp distribution per object across the dataset"""
    print(f"Starting grasp distribution analysis...")
    
    # If sample_size is specified, analyze only a subset of samples
    total_samples = len(dataset)
    if sample_size and sample_size < total_samples:
        indices = np.random.choice(total_samples, sample_size, replace=False)
        print(f"Randomly sampling {sample_size} samples for grasp analysis (total {total_samples} samples)")
    else:
        indices = range(total_samples)
        print(f"Analyzing grasps in all {total_samples} samples")
    
    object_grasp_counts = Counter()  # Count total grasps per object across all samples
    object_sample_counts = Counter()  # Count how many samples each object appears in
    sample_grasp_counts = []  # Count of grasp-containing objects per sample
    sample_total_grasps = []  # Total grasp count per sample
    
    start_time = time.time()
    
    for i, idx in enumerate(indices):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(indices) - i) / rate
            print(f"Progress: {i}/{len(indices)} ({i/len(indices)*100:.1f}%) - "
                  f"Speed: {rate:.1f} samples/s - Estimated remaining: {remaining:.1f}s")
        
        try:
            # Get tokens from dataset
            tokens, max_seq_len, scene_grasps = dataset[idx]
            tokens = tokens.squeeze()  # Remove batch dimension if present
            # Convert tokens to list if it's a tensor
            if isinstance(tokens, torch.Tensor):
                # Convert tensor to mapping first, then decode to get original tokens
                token_ids = tokens.tolist()
                # Get inverse mapping from dataset
                mapping = dataset.token_mapping
                inv_mapping = {v: k for k, v in mapping.items()}
                token_list = [inv_mapping[tid] for tid in token_ids if tid in inv_mapping]
            else:
                token_list = tokens
            
            # Parse tokens to get AST
            try:
                parser = Parser(token_list)
                ast = parser.parse()
            except Exception as parse_error:
                print(f"Failed to parse tokens in sample {idx}: {parse_error}")
                continue
            
            # Count objects and grasps for each object in this sample
            sample_object_grasps = Counter()
            sample_objects_present = set()  # All objects present in this sample
            total_grasps_in_sample = 0
            
            # Traverse AST to find all objects and grasps
            for item in ast.items:
                if isinstance(item, SB):  # Labeled segment - object present in scene
                    sample_objects_present.add(item.tag)
                elif isinstance(item, GRASP):  # Grasp actions
                    for gb in item.gbs:  # Each GB represents one grasp
                        object_tag = gb.tag
                        sample_object_grasps[object_tag] += 1
                        total_grasps_in_sample += 1
                        sample_objects_present.add(object_tag)  # Grasped objects are also present
            
            # Update global counters
            object_grasp_counts.update(sample_object_grasps)
            # Update sample counts for all objects present in this sample
            for obj_tag in sample_objects_present:
                object_sample_counts[obj_tag] += 1
                
            sample_grasp_counts.append(len(sample_object_grasps))  # Number of different objects with grasps
            sample_total_grasps.append(total_grasps_in_sample)     # Total grasp count in this sample
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Grasp analysis completed, time elapsed: {elapsed_time:.2f}s")
    
    return object_grasp_counts, object_sample_counts, np.array(sample_grasp_counts), np.array(sample_total_grasps)


def print_grasp_statistics(object_grasp_counts, object_sample_counts, sample_grasp_counts, sample_total_grasps):
    """Print detailed grasp statistics"""
    print(f"\n=== Grasp Distribution Analysis ===")
    
    if len(object_grasp_counts) == 0:
        print("No grasp information found in the analyzed samples")
        return
    
    # Overall statistics
    total_objects_with_grasps = len(object_grasp_counts)
    total_grasps_across_dataset = sum(object_grasp_counts.values())
    
    print(f"Total unique objects with grasps: {total_objects_with_grasps}")
    print(f"Total grasps across all samples: {total_grasps_across_dataset}")
    
    # Per-object grasp counts
    print(f"\n=== Grasps per Object ===")
    sorted_objects = object_grasp_counts.most_common()
    
    grasp_counts_per_object = [count for _, count in sorted_objects]
    print(f"Average grasps per object: {np.mean(grasp_counts_per_object):.2f}")
    print(f"Median grasps per object: {np.median(grasp_counts_per_object):.2f}")
    print(f"Min grasps per object: {np.min(grasp_counts_per_object)}")
    print(f"Max grasps per object: {np.max(grasp_counts_per_object)}")
    
    print(f"\nTop 10 objects with most grasps:")
    for obj, count in sorted_objects[:10]:
        print(f"  {obj}: {count} grasps")
    
    print(f"\nBottom 10 objects with fewest grasps:")
    for obj, count in sorted_objects[-10:]:
        print(f"  {obj}: {count} grasps")
    
    # Average grasps per object in samples where it appears
    print(f"\n=== Average Grasps per Object in Samples Where It Appears ===")
    print(f"Total objects present in dataset: {len(object_sample_counts)}")
    
    # Calculate average grasps per object per sample where it appears
    object_avg_grasps = []
    for obj in object_grasp_counts:
        if obj in object_sample_counts and object_sample_counts[obj] > 0:
            avg_grasps = object_grasp_counts[obj] / object_sample_counts[obj]
            object_avg_grasps.append((obj, avg_grasps, object_grasp_counts[obj], object_sample_counts[obj]))
    
    # Sort by average grasps per sample
    object_avg_grasps.sort(key=lambda x: x[1], reverse=True)
    
    if len(object_avg_grasps) > 0:
        avg_values = [x[1] for x in object_avg_grasps]
        print(f"Average grasps per object per sample (across all objects): {np.mean(avg_values):.2f}")
        print(f"Median grasps per object per sample: {np.median(avg_values):.2f}")
        print(f"Min average grasps per object per sample: {np.min(avg_values):.2f}")
        print(f"Max average grasps per object per sample: {np.max(avg_values):.2f}")
        
        print(f"\nTop 10 objects with highest average grasps per sample:")
        for obj, avg_grasps, total_grasps, sample_count in object_avg_grasps[:10]:
            print(f"  {obj}: {avg_grasps:.2f} grasps/sample ({total_grasps} total grasps in {sample_count} samples)")
        
        print(f"\nBottom 10 objects with lowest average grasps per sample:")
        for obj, avg_grasps, total_grasps, sample_count in object_avg_grasps[-10:]:
            print(f"  {obj}: {avg_grasps:.2f} grasps/sample ({total_grasps} total grasps in {sample_count} samples)")
    
    # Per-sample statistics
    print(f"\n=== Per-Sample Grasp Statistics ===")
    valid_samples = sample_total_grasps[sample_total_grasps > 0]  # Only samples with grasps
    print(f"Samples containing grasps: {len(valid_samples)} out of {len(sample_total_grasps)} ({len(valid_samples)/len(sample_total_grasps)*100:.1f}%)")
    
    if len(valid_samples) > 0:
        print(f"Average grasps per sample (with grasps): {np.mean(valid_samples):.2f}")
        print(f"Median grasps per sample (with grasps): {np.median(valid_samples):.2f}")
        print(f"Min grasps per sample: {np.min(valid_samples)}")
        print(f"Max grasps per sample: {np.max(valid_samples)}")
        
        print(f"Average objects with grasps per sample: {np.mean(sample_grasp_counts[sample_grasp_counts > 0]):.2f}")
    
    # Grasp count distribution
    print(f"\n=== Grasp Count Distribution ===")
    unique_counts, count_frequencies = np.unique(grasp_counts_per_object, return_counts=True)
    
    print("Grasp count histogram (objects with N grasps):")
    for count, freq in zip(unique_counts, count_frequencies):
        percentage = freq / len(grasp_counts_per_object) * 100
        print(f"  {count} grasps: {freq} objects ({percentage:.1f}%)")


def plot_grasp_distribution(object_grasp_counts, sample_total_grasps):
    """Plot grasp distribution visualizations"""
    if len(object_grasp_counts) == 0:
        print("No grasp data to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Grasps per object distribution
    plt.subplot(2, 3, 1)
    grasp_counts = [count for _, count in object_grasp_counts.items()]
    plt.hist(grasp_counts, bins=min(50, max(grasp_counts)), alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Grasps')
    plt.ylabel('Number of Objects')
    plt.title('Distribution of Grasps per Object')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Top objects with most grasps
    plt.subplot(2, 3, 2)
    top_objects = object_grasp_counts.most_common(20)
    if len(top_objects) > 0:
        objects, counts = zip(*top_objects)
        y_pos = np.arange(len(objects))
        plt.barh(y_pos, counts)
        plt.yticks(y_pos, [obj[:10] + '...' if len(obj) > 10 else obj for obj in objects])
        plt.xlabel('Number of Grasps')
        plt.title('Top 20 Objects by Grasp Count')
        plt.tight_layout()
    
    # Plot 3: Grasps per sample distribution  
    plt.subplot(2, 3, 3)
    valid_samples = sample_total_grasps[sample_total_grasps > 0]
    if len(valid_samples) > 0:
        plt.hist(valid_samples, bins=min(50, max(valid_samples)), alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Grasps per Sample')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Grasps per Sample')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution
    plt.subplot(2, 3, 4)
    sorted_counts = np.sort(grasp_counts)
    y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, y, linewidth=2)
    plt.xlabel('Number of Grasps per Object')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Grasps per Object')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Grasp count frequency
    plt.subplot(2, 3, 5)
    unique_counts, frequencies = np.unique(grasp_counts, return_counts=True)
    plt.bar(unique_counts, frequencies, alpha=0.7)
    plt.xlabel('Number of Grasps')
    plt.ylabel('Number of Objects')
    plt.title('Frequency of Different Grasp Counts')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Log-scale view for better visibility
    plt.subplot(2, 3, 6)
    plt.hist(grasp_counts, bins=min(50, max(grasp_counts)), alpha=0.7, edgecolor='black')
    plt.yscale('log')
    plt.xlabel('Number of Grasps per Object')
    plt.ylabel('Number of Objects (log scale)')
    plt.title('Grasp Distribution (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grasp_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Grasp distribution plot saved: grasp_distribution_analysis.png")
    plt.close()

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
    
    # Analyze grasp distribution
    print(f"\n" + "="*60)
    print(f"Starting grasp analysis...")
    object_grasp_counts, object_sample_counts, sample_grasp_counts, sample_total_grasps = analyze_grasp_distribution(dataset, sample_size)
    
    # Print grasp statistics
    print_grasp_statistics(object_grasp_counts, object_sample_counts, sample_grasp_counts, sample_total_grasps)
    
    # Plot grasp distributions
    plot_grasp_distribution(object_grasp_counts, sample_total_grasps)
    
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
    bins = [0, 100, 500, 1000, 1500, 2000, 3000, 4000,  max_seq_len, np.inf]
    labels = ['<100', '100-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000-4000', f'4000-{max_seq_len}', f'>{max_seq_len}']
    
    hist, _ = np.histogram(lengths, bins=bins)
    for i, (label, count) in enumerate(zip(labels, hist)):
        percentage = count / len(lengths) * 100
        print(f"{label:>12}: {count:>6} samples ({percentage:>5.1f}%)")

if __name__ == "__main__":
    main()