import time
import logging
import numpy as np
from numba import jit, prange
from typing import Dict


@jit(nopython=True)
def _prune_infrequent_1_sequences(
    candidates: np.ndarray, minsup: int, sequences: np.ndarray
) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty(0, dtype=candidates.dtype)

    # count support for each candidate
    support_counts = np.zeros(len(candidates), dtype=np.int64)
    for i in range(len(candidates)):
        # For 1-sequences, we reshape to make it compatible with _count_support_parallel
        candidate_seq = np.array([candidates[i]])
        support_counts[i] = _count_support_parallel(candidate_seq, sequences)

    frequent_indices = np.where(support_counts >= minsup)[0]
    frequent_sequences = np.empty(len(frequent_indices), dtype=candidates.dtype)

    # Copy frequent sequences
    for i in range(len(frequent_indices)):
        frequent_sequences[i] = candidates[frequent_indices[i]]

    return frequent_sequences


@jit(nopython=True)
def _prune_infrequent_k_sequences(
    candidates: np.ndarray, minsup: int, sequences: np.ndarray
) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty((0, candidates.shape[1]), dtype=candidates.dtype)

    support_counts = np.zeros(len(candidates), dtype=np.int64)
    for i in range(len(candidates)):
        support_counts[i] = _count_support_parallel(candidates[i], sequences)

    frequent_indices = np.where(support_counts >= minsup)[0]
    frequent_sequences = np.empty(
        (len(frequent_indices), candidates.shape[1]), dtype=candidates.dtype
    )

    # Copy frequent sequences
    for i in range(len(frequent_indices)):
        frequent_sequences[i] = candidates[frequent_indices[i]]

    return frequent_sequences


@jit(nopython=True)
def _is_subsequence(target: np.ndarray, sequence: np.ndarray) -> bool:
    """Check if target is a subsequence of sequence"""
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
    for i in prange(len(sequences)):
        if _is_subsequence(candidate, sequences[i]):
            support += 1

    return support


@jit(nopython=True)
def _get_unique_events(sequences: np.ndarray) -> np.ndarray:
    total_len = 0
    for i in range(len(sequences)):
        total_len += len(sequences[i])

    # Create and fill flat array
    flat = np.empty(total_len, dtype=sequences.dtype)
    idx = 0
    for i in range(len(sequences)):
        seq = sequences[i]
        for j in range(len(seq)):
            flat[idx] = seq[j]
            idx += 1

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
    assert freq_1_seq.ndim == 1

    n = len(freq_1_seq)
    # Total number of 2-sequences will be n * n
    # (all possible pairs, including same item)
    total_sequences = n * n

    # Pre-allocate output array
    candidates = np.empty((total_sequences, 2), dtype=freq_1_seq.dtype)

    # Generate all possible pairs
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

    # Count exact number of valid joins first
    n_candidates = count_valid_joins(freq_k_seq)

    # Allocate exact size needed
    candidates = np.empty((n_candidates, k + 1), dtype=freq_k_seq.dtype)

    # Generate candidates
    idx = 0
    for i in range(n_sequences):
        for j in range(n_sequences):
            if _sequences_can_join(freq_k_seq[i], freq_k_seq[j]):
                candidates[idx] = _join_sequences(freq_k_seq[i], freq_k_seq[j])
                idx += 1

    return candidates


@jit(nopython=True)
def count_valid_joins(freq_k_seq: np.ndarray) -> int:
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
        self, sequences: np.ndarray, min_support: int, verbose: bool = False
    ) -> None:
        assert min_support > 0
        self.minsup = min_support
        self.seqdb = sequences
        self.verbose = verbose

        self.frequent_seqs: Dict[int, np.ndarray] = {}

    def count_support(self, sequence: np.ndarray) -> int:
        return _count_support_parallel(sequence, self.seqdb)

    def find_freq_seq(self) -> Dict[int, np.ndarray]:
        self.freq_seqs = {}

        start = time.time()

        # start with singletons
        candidates: np.ndarray = _get_unique_events(self.seqdb)  # shape (N,)
        freq_1_seq = _prune_infrequent_1_sequences(candidates, self.minsup, self.seqdb)
        self.freq_seqs[1] = freq_1_seq
        if self.verbose:
            logging.info(f"Generated freq 1-sequence: {freq_1_seq}")

        # base case: 2-sequences
        k = 2
        candidates: np.ndarray = _get_base_freq_seq(freq_1_seq)  # shape (N, 2)
        freq_k_seq = _prune_infrequent_k_sequences(candidates, self.minsup, self.seqdb)
        self.freq_seqs[k] = freq_k_seq
        if self.verbose:
            logging.info(f"Generated freq 2-sequence: {freq_k_seq}")

        # iterative cases
        while freq_k_seq.shape[0] > 0:
            assert k == freq_k_seq.shape[1]
            candidates = _get_next_freq_seq(freq_k_seq)  # shape (N, k+1)
            freq_k_seq = _prune_infrequent_k_sequences(
                candidates, self.minsup, self.seqdb
            )
            assert k + 1 == freq_k_seq.shape[1]
            k += 1
            if len(freq_k_seq) == 0:
                break
            self.freq_seqs[k] = freq_k_seq
            if self.verbose:
                logging.info(f"Generated freq {k}-sequence: {freq_k_seq}")

        end = time.time()
        runtime = end - start

        if self.verbose:
            logging.info(f"Found frequence sequences in {runtime:.4f} secs")

        return self.freq_seqs

    @classmethod
    def seq_to_str(cls, seq: np.ndarray) -> str:
        assert seq.ndim == 1
        s = ",".join(sorted(str(item) for item in seq))
        return "{" + s + "}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    sequences = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 4, 5], [1, 3, 7, 2]])
    gsp = GSP(sequences, min_support=2, verbose=True)
    candidate = np.array([1, 4])
    support = gsp.count_support(candidate)
    print(gsp.find_freq_seq())
