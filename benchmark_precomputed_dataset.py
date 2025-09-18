#!/usr/bin/env python3
"""
Benchmark script to test PrecomputedDataset throughput
Tests data loading speed with different batch sizes and num_workers
"""

import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graspGPT.model.precomputed_dataset import PrecomputedDataset
import numpy as np
from tqdm import tqdm

def benchmark_dataset(data_path: str, 
                     batch_sizes: list = [1, 4, 8],
                     num_workers_list: list = [0, 2, 4],
                     num_samples: int = 100,
                     max_sequence_length: int = 5024):
    """
    Benchmark PrecomputedDataset with different configurations
    
    Args:
        data_path: Path to precomputed data directory or file
        batch_sizes: List of batch sizes to test
        num_workers_list: List of num_workers values to test
        num_samples: Number of samples to process (or full dataset if smaller)
        max_sequence_length: Maximum sequence length
    """
    
    print(f"Loading PrecomputedDataset from: {data_path}")
    dataset = PrecomputedDataset(data_path, max_sequence_length=max_sequence_length)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    print(f"Volume dimensions: {dataset.volume_dims}")
    print("-" * 80)
    
    # Limit samples to test if dataset is large
    actual_samples = min(num_samples, len(dataset))
    
    results = []
    
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"Testing batch_size={batch_size}, num_workers={num_workers}")
            
            # Custom collate function to handle None values
            def custom_collate(batch):
                tokens, seq_lengths, scene_grasps = zip(*batch)
                # Stack tokens and seq_lengths, ignore scene_grasps (which are None)

                return tokens, seq_lengths, None
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                collate_fn=custom_collate
            )
            
            # Warmup
            warmup_batches = 3
            for i, batch in enumerate(dataloader):
                if i >= warmup_batches:
                    break
            
            # Benchmark
            start_time = time.time()
            processed_samples = 0
            
            with tqdm(desc=f"B{batch_size}_W{num_workers}", 
                     total=actual_samples, 
                     unit="samples") as pbar:
                
                for batch_idx, (tokens, seq_lengths, scene_grasps) in enumerate(dataloader):
                    batch_samples = len(tokens)
                    processed_samples += batch_samples
                    pbar.update(batch_samples)
                    
                    # Stop when we've processed enough samples
                    if processed_samples >= actual_samples:
                        break
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate metrics
            samples_per_second = processed_samples / elapsed_time
            batches_per_second = (batch_idx + 1) / elapsed_time
            
            result = {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'samples_processed': processed_samples,
                'elapsed_time': elapsed_time,
                'samples_per_second': samples_per_second,
                'batches_per_second': batches_per_second
            }
            results.append(result)
            
            print(f"  Processed {processed_samples} samples in {elapsed_time:.2f}s")
            print(f"  Throughput: {samples_per_second:.2f} samples/sec, {batches_per_second:.2f} batches/sec")
            print()
    
    # Print summary
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Workers':<8} {'Samples/sec':<12} {'Batches/sec':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['batch_size']:<12} {result['num_workers']:<8} "
              f"{result['samples_per_second']:<12.2f} {result['batches_per_second']:<12.2f} "
              f"{result['elapsed_time']:<10.2f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['samples_per_second'])
    print("-" * 80)
    print(f"Best throughput: {best_result['samples_per_second']:.2f} samples/sec")
    print(f"Best configuration: batch_size={best_result['batch_size']}, num_workers={best_result['num_workers']}")
    
    return results

def test_single_sample_speed(data_path: str, num_tests: int = 100):
    """
    Test speed of getting individual samples (without DataLoader)
    """
    print(f"\nTesting individual sample access speed...")
    
    dataset = PrecomputedDataset(data_path)
    
    # Random indices to test
    indices = np.random.choice(len(dataset), size=min(num_tests, len(dataset)), replace=False)
    
    start_time = time.time()
    for idx in tqdm(indices, desc="Single samples"):
        tokens, seq_len, scene_grasps = dataset[idx]
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    samples_per_second = len(indices) / elapsed_time
    
    print(f"Individual sample access: {samples_per_second:.2f} samples/sec")
    print(f"Average time per sample: {elapsed_time/len(indices)*1000:.2f}ms")
    
    return samples_per_second

def main():
    parser = argparse.ArgumentParser(description='Benchmark PrecomputedDataset throughput')
    parser.add_argument('--data_path', default= "output/precomputed_data", help='Path to precomputed data directory or file')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8],
                       help='Batch sizes to test (default: 1 4 8 16 32)')
    parser.add_argument('--num-workers', nargs='+', type=int, default=[0,  2, 8],
                       help='Number of workers to test (default: 0 1 2 4)')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to process per test (default: 200)')
    parser.add_argument('--max-seq-len', type=int, default=5000,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--test-single', action='store_true',
                       help='Also test individual sample access speed')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist")
        return
    
    print("PrecomputedDataset Throughput Benchmark")
    print("=" * 80)
    
    # Main benchmark
    results = benchmark_dataset(
        data_path=str(data_path),
        batch_sizes=args.batch_sizes,
        num_workers_list=args.num_workers,
        num_samples=args.num_samples,
        max_sequence_length=args.max_seq_len
    )
    
    # Optional single sample test
    if args.test_single:
        test_single_sample_speed(str(data_path))
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()