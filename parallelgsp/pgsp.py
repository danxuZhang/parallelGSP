#!/usr/bin/env python3
import time
import logging
import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple


def _list_to_ndarray(sequences: List[List[int]]) -> Tuple[np.ndarray, int]:
    """Convert List of sequences of different lengths to [N, MaxK] np array"""
    if not sequences:
        raise ValueError("Input sequences cannot be empty")

    # validate input values
    for seq in sequences:
        if not seq:  # Check for empty sequences
            raise ValueError("Found empty sequence")
        if any(not isinstance(x, int) or x <= 0 for x in seq):
            raise ValueError("All events must be positive integers")

    # find maximum sequence length
    max_len = max(len(seq) for seq in sequences)

    # initialize padded array with zeros
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int64)

    # fill the array with sequences
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq

    return padded_sequences, 0  # always pad with 0


def _to_support_dict(
    sequences: np.ndarray, supports: np.ndarray
) -> Dict[Tuple[int], int]:
    """
    Convert sequences and support arrays to a dictionary mapping sequences
    to their support counts
    """
    support_dict = {}
    for i in range(len(sequences)):
        # Convert sequence to tuple so it can be used as dictionary key
        if sequences[i].ndim == 0:  # scalar value
            key = int(sequences[i])  # for 1-sequences
        else:
            # Convert to tuple and remove padding zeros
            key = tuple(int(x) for x in sequences[i] if x != 0)
        support_dict[key] = supports[i]
    return support_dict


@jit(nopython=True)
def _prune_infrequent_1_sequences(
    candidates: np.ndarray, minsup: int, sequences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Prune infrequent 1-sequences"""
    if len(candidates) == 0:
        return np.empty(0, dtype=candidates.dtype), np.empty(0, dtype=np.int64)

    # count support for each candidate
    support_counts = np.zeros(len(candidates), dtype=np.int64)
    for i in range(len(candidates)):
        # for 1-sequences, we reshape to make it compatible with _count_support_parallel
        candidate_seq = np.array([candidates[i]])
        support_counts[i] = _count_support_parallel(candidate_seq, sequences)

    frequent_indices = np.where(support_counts >= minsup)[0]
    frequent_sequences = np.empty(len(frequent_indices), dtype=candidates.dtype)
    frequent_supports = np.empty(len(frequent_indices), dtype=np.int64)

    # copy frequent sequences
    for i in range(len(frequent_indices)):
        idx = frequent_indices[i]
        frequent_sequences[i] = candidates[idx]
        frequent_supports[i] = support_counts[idx]

    return frequent_sequences, frequent_supports


@jit(nopython=True)
def _prune_infrequent_k_sequences(
    candidates: np.ndarray, minsup: int, sequences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Prune infrequent k-sequences (k>1)"""
    if len(candidates) == 0:
        return np.empty((0, candidates.shape[1]), dtype=candidates.dtype), np.empty(
            0, dtype=np.int64
        )

    support_counts = np.zeros(len(candidates), dtype=np.int64)
    for i in range(len(candidates)):
        support_counts[i] = _count_support_parallel(candidates[i], sequences)

    frequent_indices = np.where(support_counts >= minsup)[0]
    frequent_sequences = np.empty(
        (len(frequent_indices), candidates.shape[1]), dtype=candidates.dtype
    )
    frequent_supports = np.empty(len(frequent_indices), dtype=np.int64)

    # Copy frequent sequences
    for i in range(len(frequent_indices)):
        idx = frequent_indices[i]
        frequent_sequences[i] = candidates[idx]
        frequent_supports[i] = support_counts[idx]

    return frequent_sequences, frequent_supports


@jit(nopython=True)
def _is_subsequence(
    target: np.ndarray, sequence: np.ndarray, padding_value: int = 0
) -> bool:
    """Check if target is a subsequence of sequence"""
    # remove padding
    target = target[target != padding_value]
    sequence = sequence[sequence != padding_value]
    m, n = len(target), len(sequence)
    if m > n:
        return False

    i = 0  # index for target
    j = 0  # index for sequence

    while i < m and j < n:
        if target[i] == sequence[j]:
            i += 1
        j += 1

    return i == m


@jit(nopython=True, parallel=True)
def _count_support_parallel(candidate: np.ndarray, sequences: np.ndarray) -> int:
    """Count support for a candidate in parallel"""
    support = 0

    # Parallel loop over sequences
    for i in prange(sequences.shape[0]):
        if _is_subsequence(candidate, sequences[i]):
            support += 1

    return support


@jit(nopython=True)
def _get_unique_events(sequences: np.ndarray) -> np.ndarray:
    """Get unique events excluding padding value (0)"""
    total_len = 0
    for i in range(len(sequences)):
        total_len += len(sequences[i])

    # Create and fill flat array
    flat = np.empty(total_len, dtype=sequences.dtype)
    idx = 0
    for i in range(len(sequences)):
        seq = sequences[i]
        for j in range(len(seq)):
            if seq[j] != 0:  # Skip padding values
                flat[idx] = seq[j]
                idx += 1

    # Trim the flat array to actual size used
    flat = flat[:idx]
    return np.unique(flat)


@jit(nopython=True)
def _sequences_can_join(seq1: np.ndarray, seq2: np.ndarray) -> bool:
    """
    Check if two sequences can be joined by verifying:
    The subsequence obtained by removing an event from the first element in seq1
    is the same as what obtained by removing an event from the last element in seq
    """
    assert seq1.ndim == 1
    assert seq2.ndim == 1
    assert len(seq1) == len(seq2)

    # cannot join two identical array
    if np.array_equal(seq1, seq2):
        return False

    return np.array_equal(seq1[1:], seq2[:-1])


@jit(nopython=True)
def _join_sequences(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    """
    Join two sequences according to the rules:
    If last element of seq2 has only one event, append it to seq1 as new element (always)
    Otherwise add the missing event from seq2's last element to seq1's last element
    """
    k = len(seq1)
    result = np.empty(k + 1, dtype=seq1.dtype)
    result[:-1] = seq1  # copy all elements from seq1
    result[-1] = seq2[-1]  # append last element from seq2
    return result


@jit(nopython=True)
def _get_base_freq_seq(freq_1_seq: np.ndarray) -> np.ndarray:
    """Generate candidate 2-sequences from 1-sequences"""
    assert freq_1_seq.ndim == 1

    n = len(freq_1_seq)
    # total number of 2-sequences will be n * n
    # (all possible pairs, including same item)
    total_sequences = n * n

    # pre-allocate output array
    candidates = np.empty((total_sequences, 2), dtype=freq_1_seq.dtype)

    # generate all possible pairs
    idx = 0
    for i in range(n):
        for j in range(n):
            candidates[idx, 0] = freq_1_seq[i]
            candidates[idx, 1] = freq_1_seq[j]
            idx += 1

    return candidates


@jit(nopython=True)
def _get_next_freq_seq(freq_k_seq: np.ndarray) -> np.ndarray:
    """Generate candidate (k+1)-sequences from k-sequences."""
    assert freq_k_seq.ndim == 2, "Input must be 2D array"
    n_sequences = len(freq_k_seq)
    k = freq_k_seq.shape[1]

    # count exact number of valid joins first
    n_candidates = _count_valid_joins(freq_k_seq)

    # allocate for possible joins
    candidates = np.empty((n_candidates, k + 1), dtype=freq_k_seq.dtype)

    # generate candidates
    idx = 0
    for i in range(n_sequences):
        for j in range(n_sequences):
            if _sequences_can_join(freq_k_seq[i], freq_k_seq[j]):
                candidates[idx] = _join_sequences(freq_k_seq[i], freq_k_seq[j])
                idx += 1

    return candidates


@jit(nopython=True)
def _count_valid_joins(freq_k_seq: np.ndarray) -> int:
    """Count number of valid joins to pre-allocate exact size needed."""
    n_sequences = len(freq_k_seq)
    count = 0
    for i in range(n_sequences):
        for j in range(n_sequences):
            if np.array_equal(freq_k_seq[i][1:], freq_k_seq[j][:-1]):
                if not np.array_equal(freq_k_seq[i], freq_k_seq[j]):
                    count += 1
    return count


class GSP:
    def __init__(
        self, sequences: List[List[int]], min_support: int, verbose: bool = False
    ) -> None:
        assert min_support > 0
        self.minsup = min_support
        self.seqdb, self.padding = _list_to_ndarray(sequences)
        assert self.padding == 0  # use 0 as padding value
        self.verbose = verbose

        self.frequent_seqs: Dict[int, np.ndarray] = {}
        self.frequent_sups: List[Dict[Tuple[int], int]] = []

    def count_support(self, sequence: List[int]) -> int:
        """Count Support for a sequence"""
        # Pad sequence to match database sequence length
        padded_seq = np.zeros(
            self.seqdb.shape[1], dtype=np.int64
        )  # use same padding as seqdb
        padded_seq[: len(sequence)] = sequence
        return _count_support_parallel(padded_seq, self.seqdb)

    def find_freq_seq(
        self,
    ) -> Tuple[Dict[int, np.ndarray], List[Dict[Tuple[int], int]]]:
        """Find all frequent sequences"""
        self.frequent_seqs = {}

        if self.verbose:
            logging.info("Start finding frequent sequences...")
        start = time.time()

        # start with singletons
        candidates: np.ndarray = _get_unique_events(self.seqdb)  # shape (N,)
        if self.verbose:
            logging.info(f"Found {len(candidates)} candidate 1-sequences...")
        freq_1_seq, freq_1_sup = _prune_infrequent_1_sequences(
            candidates, self.minsup, self.seqdb
        )
        self.frequent_sups.append(_to_support_dict(freq_1_seq, freq_1_sup))
        self.frequent_seqs[1] = freq_1_seq
        if self.verbose:
            logging.info(f"Found {len(freq_1_seq)} freq 1-sequences")

        # base case: 2-sequences
        k = 2
        candidates: np.ndarray = _get_base_freq_seq(freq_1_seq)  # shape (N, 2)
        if self.verbose:
            logging.info(f"Found {len(candidates)} candidate 2-sequences...")
        freq_k_seq, freq_k_sup = _prune_infrequent_k_sequences(
            candidates, self.minsup, self.seqdb
        )
        self.frequent_sups.append(_to_support_dict(freq_k_seq, freq_k_sup))
        self.frequent_seqs[k] = freq_k_seq
        assert 2 == freq_k_seq.shape[1]
        if self.verbose:
            logging.info(f"Found {len(freq_k_seq)} freq 2-sequences")

        # iterative cases
        while freq_k_seq.shape[0] > 0:
            assert k == freq_k_seq.shape[1]
            candidates = _get_next_freq_seq(freq_k_seq)  # shape (N, k+1)
            if self.verbose:
                logging.info(f"Found {len(candidates)} candidate {k}-sequences...")
            freq_k_seq, freq_k_sup = _prune_infrequent_k_sequences(
                candidates, self.minsup, self.seqdb
            )
            assert k + 1 == freq_k_seq.shape[1]
            k += 1
            if len(freq_k_seq) == 0 and self.verbose:
                logging.info(f"No frequent {k}-sequence")
                break

            self.frequent_seqs[k] = freq_k_seq
            self.frequent_sups.append(_to_support_dict(freq_k_seq, freq_k_sup))
            if self.verbose:
                logging.info(f"Found {len(freq_k_seq)} freq {k}-sequences")

        end = time.time()
        runtime = end - start

        if self.verbose:
            logging.info(f"Found all frequent sequences in {runtime:.4f} secs")

        return self.frequent_seqs, self.frequent_sups

    def warmup(self):
        """Warmup to compile all JIT functions"""
        if self.verbose:
            logging.info("Warmup to compile JIT functions")
        minsup = 2
        sequences = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 4, 5]]
        gsp = GSP(sequences, min_support=minsup, verbose=False)
        _ = gsp.count_support([1, 2])
        _ = gsp.find_freq_seq()
        if self.verbose:
            logging.info("Warmup finished")

    def export_to_file(self, file_name: str):
        """
        Export frequent sequence with support to file
        Format: "sup seq" in each line
        Example: "10 1 3 2" - sequence {1, 3, 2} has support 10
        """
        if self.verbose:
            logging.info(f"Writing frequent sequences with support to {file_name}")
        cnt = 0
        start = time.time()
        with open(file_name, "w") as f:
            for length in sorted(self.frequent_seqs.keys()):
                sequences = self.frequent_seqs[length]
                supports: Dict[Tuple[int], int] = self.frequent_sups[length - 1]

                for seq in sequences:
                    if length == 1:
                        key: int = int(seq)  # for 1-sequences
                        f.write(f"{supports[(key)]} {key}\n")
                    else:
                        # Remove padding zeros and convert to tuple
                        key: Tuple[int] = tuple(int(x) for x in seq if x != 0)
                        pattern = " ".join(str(x) for x in key)
                        f.write(f"{supports[key]} {pattern}\n")
                    cnt += 1

        end = time.time()
        if self.verbose:
            logging.info(f"Wrote {cnt} records to {file_name} in {end-start:.4f} secs")

    @classmethod
    def seq_to_str(cls, seq: np.ndarray) -> str:
        assert seq.ndim == 1
        s = ",".join(str(item) for item in seq)
        return "{" + s + "}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    sequences = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 4, 5], [1, 3, 7], [3, 1]]
    gsp = GSP(sequences, min_support=3, verbose=True)
    candidate = [1, 4]
    support = gsp.count_support(candidate)
    seq, sup = gsp.find_freq_seq()
    print(seq)
    print(sup)
    # gsp.export_to_file("output.txt")
