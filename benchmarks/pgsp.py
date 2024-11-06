import argparse
import logging
from typing import List
import time

from parallelgsp import GSP


def run_gsp_analysis(
    input_sequences: List[List[int]],
    min_support: float,
    output_file: str,
    verbose: bool = True,
) -> None:
    """
    Run GSP analysis on the sequences and write results to output file
    """
    # Initialize GSP
    gsp = GSP(
        input_sequences,
        min_support=int(min_support * len(input_sequences)),
        verbose=verbose,
    )

    # Warm up JIT functions
    gsp.warmup()

    # Find frequent sequences
    if verbose:
        print(f"\nFinding frequent sequences with minimum support = {min_support}")
    freq_seqs = gsp.find_freq_seq()

    # Write results to file
    with open(output_file, "w") as f:
        # Write results in specified format
        for k, sequences in freq_seqs.items():
            if k == 1:
                # Write 1-sequences
                for seq in sequences:
                    support = gsp.count_support([seq])
                    f.write(f"{seq} -1 #SUP: {support}\n")
            else:
                # Write k-sequences
                for seq in sequences:
                    support = gsp.count_support(seq)
                    pattern = " -1 ".join(str(x) for x in seq[seq != 0])
                    f.write(f"{pattern} -1 #SUP: {support}\n")


def load_sequences(filepath: str) -> List[List[int]]:
    """Load sequences from input file"""
    sequences = []
    current_sequence = []

    with open(filepath, "r") as f:
        for line in f:
            # Split line into items and convert to integers
            items = [int(x) for x in line.strip().split()]
            sequence = []
            for item in items:
                if item == -2:  # End of sequence
                    if sequence:
                        sequences.append(sequence)
                    sequence = []
                elif item != -1:  # Skip item delimiters (-1)
                    sequence.append(item)
            if sequence:  # Add last sequence if exists
                sequences.append(sequence)

    return sequences


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run GSP algorithm on sequence data")
    parser.add_argument("input_file", type=str, help="Input file containing sequences")
    parser.add_argument(
        "output_file",
        type=str,
        help="Output file for frequent sequences",
    )
    parser.add_argument(
        "--min_support_ratio",
        type=float,
        default=0.25,
        help="Minimum support threshold",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # Load sequences
    if args.verbose:
        print(f"Loading sequences from {args.input_file}")
    sequences = load_sequences(args.input_file)

    if args.verbose:
        print(f"Loaded {len(sequences)} sequences")

    # Run analysis
    start_time = time.time()
    run_gsp_analysis(sequences, args.min_support_ratio, args.output_file, args.verbose)
    end_time = time.time()

    if args.verbose:
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
