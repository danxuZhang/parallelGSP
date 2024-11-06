from parallelgsp import GSP

import numpy as np
from typing import List, Tuple, Dict
import logging
import time

def load_fifa_data(filepath: str) -> List[List[int]]:
    """
    Load FIFA dataset and convert it to list of sequences.
    Each sequence is a list of integers, where -1 and -2 are special delimiters.
    """
    sequences = []
    current_sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Split line into items and convert to integers
            items = [int(x) for x in line.strip().split()]
            
            # Process items
            current_seq = []
            for item in items:
                if item == -2:  # End of sequence
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = []
                elif item != -1:  # Skip item delimiters (-1)
                    current_seq.append(item)
                    
            if current_seq:  # Add last sequence if exists
                sequences.append(current_seq)
                
    return sequences

def run_gsp_analysis(input_sequences: List[List[int]], min_support: int, verbose: bool = True) -> None:
    """
    Run GSP analysis on the sequences and print results
    """
    # Initialize GSP
    gsp = GSP(input_sequences, min_support=min_support, verbose=verbose)
    
    # Warm up JIT functions
    gsp.warmup()
    
    # Find frequent sequences
    print(f"\nFinding frequent sequences with minimum support = {min_support}")
    start = time.time()
    freq_seqs = gsp.find_freq_seq()
    end = time.time()
    # Print results
    total_patterns = 0
    for k, sequences in freq_seqs.items():
        num_patterns = len(sequences)
        total_patterns += num_patterns
        print(f"\nFrequent {k}-sequences ({num_patterns} patterns):")
        if num_patterns > 0:
            for seq in sequences:
                # Convert sequence to appropriate numpy array format
                if seq.ndim == 1:
                    seq_array = seq
                else:
                    # Handle case where sequence might be flattened
                    seq_array = seq.reshape(-1)
                
                support = gsp.count_support(seq_array)
                print(f"Pattern {gsp.seq_to_str(seq_array)}: support = {support}")
    
    print(f"\nTotal number of frequent patterns found: {total_patterns}")
    print(f"\nTotal Runtime: {end-start:.4f} secs")


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load FIFA dataset
    filepath = "FIFA.txt"  # Adjust path as needed
    sequences = load_fifa_data(filepath)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Run GSP with different minimum support thresholds
    min_support = int(0.20 * len(sequences))  # Adjust these values based on your needs
    
    run_gsp_analysis(sequences, min_support)

if __name__ == "__main__":
    main()



