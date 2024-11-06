import time
from typing import List, Set


Itemset = Set[int]
Sequence = List[Itemset]
SequenceDB = List[Sequence]


class GSP:
    def __init__(
        self,
        transactions: SequenceDB,
        min_support: int,
        verbose: bool = True,
    ) -> None:
        self.seqdb = transactions
        self.min_sup = min_support
        self.frequent_patterns: List[List[Sequence]] = []
        self.verbose = verbose

        # runtime statistics
        self.runtime = -1
        self.candidates = 0

        # debugging
        self.debug = False
        if self.debug:  # force verbose when debugging
            self.verbose = True

    def count_support(self, seq: Sequence) -> int:
        """Count support for a sequence in the transaction DB"""
        self.candidates += 1  # all candidates must be counted once
        support = 0

        for transaction in self.seqdb:
            if self._is_subsequence(seq, transaction):
                support += 1

        return support

    def find_frequent_sequences(self) -> List[List[Sequence]]:
        """Find all frequent sequences"""
        self.candidates = 0
        start = time.time()

        # starts with singleton
        freq_seq_1 = self._get_freq_one_seq()
        if self.verbose:
            print("Frequent 1-Seq: ", freq_seq_1)
            print(f"Considered {self.candidates} candidates so far")
        self.frequent_patterns.append(freq_seq_1)

        # base class
        candidates_2 = self._get_base_candidates(freq_seq_1)
        freq_seq_k = self._filter_candidates_by_support(candidates_2)
        if self.verbose:
            print("Frequent 2-Seq: ", freq_seq_k)
            print(f"Considered {self.candidates} candidates so far")
        self.frequent_patterns.append(freq_seq_k)

        k = 2
        while len(freq_seq_k) > 0:
            # check all candidates are k-seqences
            if self.debug:
                for seq in freq_seq_k:
                    assert (
                        self._get_seq_len(seq) == k
                    ), f"Sequence {self.sequence_to_string(seq)} length not {k}"
            # generate k+1-sequence candidates
            candidates_kplus1 = self._get_next_candidates(freq_seq_k)
            # prune infrequent candidates
            freq_seq_k = self._filter_candidates_by_support(candidates_kplus1)
            if freq_seq_k:
                if self.verbose:
                    print(f"Frequent {k}-Seq: ", freq_seq_k)
                    print(f"Considered {self.candidates} candidates so far")
                self.frequent_patterns.append(freq_seq_k)
            k += 1
        end = time.time()
        self.runtime = end - start

        if self.verbose:
            print(f"Runtime: {self.runtime} sec")
            print(f"Considered {self.candidates} possible candidates")

        return self.frequent_patterns

    def _get_freq_one_seq(self) -> List[Sequence]:
        """Generate frequent 1-sequence"""
        unique_items = set()
        for sequence in self.seqdb:
            for itemset in sequence:
                unique_items.update(itemset)

        frequent_items = []
        for item in unique_items:
            candidate_seq = [set([item])]
            support = self.count_support(candidate_seq)
            if support >= self.min_sup:
                frequent_items.append(candidate_seq)

        return frequent_items

    def _get_base_candidates(self, freq_items: List[Sequence]) -> List[Sequence]:
        """Get frequent 2-sequence"""
        candidates = []
        items = []
        for seq in freq_items:
            if seq and seq[0]:  # Check for non-empty sequence and itemset
                items.extend(seq[0])
        items = sorted(set(items))  # Remove duplicates and sort for consistency

        # generate all combinations of two items (including same item)
        for item1 in items:
            for item2 in items:
                # Generate sequential patterns <i1, i2>
                new_seq = [set([item1]), set([item2])]
                candidates.append(new_seq)

        # generate itemset patterns <(i1,i2)> for distinct items
        for i, item1 in enumerate(items):
            for item2 in items[i + 1 :]:  # only pairs of different items
                new_seq = [set([item1, item2])]
                candidates.append(new_seq)

        return candidates

    def _get_next_candidates(self, freq_seq_k: List[Sequence]) -> List[Sequence]:
        """Generate frequent (k+1)-sequence from k-sequence (k > 1)"""
        candidates = []

        for seq1 in freq_seq_k:
            for seq2 in freq_seq_k:
                if not self._sequences_can_join(seq1, seq2):
                    continue
                new_seq = self._join_sequences(seq1, seq2)
                if self.debug:
                    k = self._get_seq_len(seq1)
                    assert (
                        self._get_seq_len(new_seq) == k + 1
                    ), f"New sequence length {new_seq} != expected {k + 1}"
                candidates.append(new_seq)

        return list(candidates)

    def _filter_candidates_by_support(
        self, candidates: List[Sequence]
    ) -> List[Sequence]:
        """Filter candidates by minimum support threshold"""
        return [
            candidate
            for candidate in candidates
            if self.count_support(candidate) >= self.min_sup
        ]

    def _get_seq_len(self, seq: Sequence) -> int:
        """Count number of events in a sequence"""
        cnt = 0
        for s in seq:
            cnt += len(s)
        return cnt

    def sequence_to_string(self, seq: Sequence) -> str:
        return ",".join(
            "{" + ",".join(sorted(str(item) for item in itemset)) + "}"
            for itemset in seq
        )

    def _are_sequences_equal(self, seq1: Sequence, seq2: Sequence) -> bool:
        """
        Check if sequences are equal using canonical string representation.
        Each itemset is sorted to ensure (a,b) and (b,a) are considered equal.
        """
        return self.sequence_to_string(seq1) == self.sequence_to_string(seq2)

    def _sequences_can_join(self, seq1: Sequence, seq2: Sequence) -> bool:
        """
        Check if two sequences can be joined by verifying:
        The subsequence obtained by removing an event from the first element in seq1
        is the same as what obtained by removing an event from the last element in seq
        """
        if self.debug:
            assert self._get_seq_len(seq1) == self._get_seq_len(seq2)

        if self._are_sequences_equal(seq1, seq2):
            return False

        first_itemset = set(seq1[0])  # create copy for modification
        last_itemset = set(seq2[-1])
        middle1 = seq1[1:]
        middle2 = seq2[:-1]

        # Check if middle sequences are equal
        if not self._are_sequences_equal(middle1, middle2):
            return False

        # Now check all possible event removals
        for e1 in first_itemset:
            temp_first = first_itemset - {e1}  # Remove one event from first
            for e2 in last_itemset:
                temp_last = last_itemset - {e2}  # Remove one event from last
                # Check if removing these events makes the sequences identical
                if temp_first == temp_last:
                    return True

        return False

    def _join_sequences(self, seq1: Sequence, seq2: Sequence) -> Sequence:
        """
        Join two sequences according to the rules:
        1. If last element of seq2 has only one event, append it to seq1 as new element
        2. Otherwise add the missing event from seq2's last element to seq1's last element
        """
        k = self._get_seq_len(seq1)
        if self.debug:
            assert k == self._get_seq_len(seq2)
            assert k > 1

        # create deep copy of seq1 to avoid modifying original
        result = [set(itemset) for itemset in seq1]

        # find which events from seq2's last itemset are missing in seq1's last itemset
        missing_events = seq2[-1] - result[-1]

        # if seq2's last itemset has exactly one event different from seq1's last itemset,
        # add it to the last itemset of result (Case 1)
        if len(missing_events) == 1 and len(seq2[-1]) > 1:
            result[-1].add(next(iter(missing_events)))
        # otherwise, append the entire last itemset as a new itemset (Case 2)
        else:
            result.append(set(seq2[-1]))

        if self.debug:
            new_len = self._get_seq_len(result)
            if new_len != k + 1:
                print(f"Failed at joining {seq1} and {seq2}, wrong result: {result}")
            assert (
                new_len == k + 1
            ), f"Joined sequence length {new_len} != expected {k + 1}"
        return result

    def _is_subsequence(self, sequence: Sequence, transaction: Sequence) -> bool:
        """Check if sequence is a subsequence of transaction [tested]"""
        if not sequence:
            return True

        if not transaction:
            return False

        if any(len(itemset) == 0 for itemset in sequence):
            return False

        pos = 0
        for s_itemset in sequence:
            found = False
            while pos < len(transaction) and not found:
                if s_itemset.issubset(transaction[pos]):
                    found = True
                pos += 1
            if not found:
                return False
        return True


if __name__ == "__main__":
    transactions = [[{1}, {2}, {3, 4}], [{1, 2}, {3}, {4}], [{1}, {2}, {3}, {4}]]
    min_support = 2
    gsp = GSP(transactions, min_support, verbose=True)
    gsp.debug = True
    frequent_sequences = gsp.find_frequent_sequences()
