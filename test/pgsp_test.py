import unittest
import numpy as np
from parallelgsp import GSP, _list_to_ndarray, _is_subsequence

class TestGSP(unittest.TestCase):
    def setUp(self):
        # Common test sequences used across multiple tests
        self.basic_sequences = [
            [1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4],
            [1, 2, 4]
        ]
        
    def test_initialization(self):
        """Test GSP class initialization"""
        gsp = GSP(self.basic_sequences, min_support=2)
        self.assertEqual(gsp.minsup, 2)
        self.assertTrue(isinstance(gsp.seqdb, np.ndarray))
        self.assertEqual(gsp.padding, 0)
        
    def test_empty_sequences(self):
        """Test handling of empty sequences"""
        with self.assertRaises(ValueError):
            GSP([], min_support=1)
            
    def test_invalid_sequences(self):
        """Test handling of invalid sequences"""
        # Test with negative numbers
        with self.assertRaises(ValueError):
            GSP([[1, -2, 3]], min_support=1)
            
        # Test with empty subsequence
        with self.assertRaises(ValueError):
            GSP([[1, 2], []], min_support=1)
            
        # Test with non-integer values
        with self.assertRaises(ValueError):
            GSP([[1, 2.5, 3]], min_support=1)
            
    def test_list_to_ndarray_conversion(self):
        """Test conversion of list sequences to numpy array"""
        sequences = [[1, 2], [1, 2, 3], [1]]
        arr, padding = _list_to_ndarray(sequences)
        
        self.assertEqual(arr.shape, (3, 3))  # 3 sequences, max length 3
        self.assertEqual(padding, 0)
        np.testing.assert_array_equal(arr[0], [1, 2, 0])
        np.testing.assert_array_equal(arr[1], [1, 2, 3])
        np.testing.assert_array_equal(arr[2], [1, 0, 0])
        
    def test_is_subsequence(self):
        """Test subsequence checking"""
        sequence = np.array([1, 2, 3, 4, 0, 0])  # padded sequence
        
        # Test various subsequences
        self.assertTrue(_is_subsequence(np.array([1, 2]), sequence))
        self.assertTrue(_is_subsequence(np.array([1, 4]), sequence))
        self.assertTrue(_is_subsequence(np.array([2, 3]), sequence))
        
        # Test non-subsequences
        self.assertFalse(_is_subsequence(np.array([4, 1]), sequence))
        self.assertFalse(_is_subsequence(np.array([1, 5]), sequence))
        self.assertFalse(_is_subsequence(np.array([1, 2, 3, 4, 5]), sequence))
        
    def test_support_counting(self):
        """Test support counting for sequences"""
        gsp = GSP(self.basic_sequences, min_support=2)
        
        # Test various patterns
        self.assertEqual(gsp.count_support(np.array([1, 2])), 3)  # appears in 3 sequences
        self.assertEqual(gsp.count_support(np.array([2, 4])), 3)  # appears in 3 sequences
        self.assertEqual(gsp.count_support(np.array([1, 4])), 2)  # appears in 2 sequences
        self.assertEqual(gsp.count_support(np.array([4, 1])), 0)  # appears in 0 sequences
        
    def test_find_freq_seq(self):
        """Test finding frequent sequences"""
        # Simple test case with clear patterns
        sequences = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 4],
            [2, 3, 4]
        ]
        gsp = GSP(sequences, min_support=2)
        freq_seqs = gsp.find_freq_seq()
        
        # Verify results
        self.assertTrue(1 in freq_seqs)  # Should have 1-sequences
        self.assertTrue(2 in freq_seqs)  # Should have 2-sequences
        
        # Check specific frequent patterns
        self.assertTrue(np.any(np.all(freq_seqs[2] == np.array([1, 2]), axis=1)))  # [1,2] should be frequent
        self.assertTrue(np.any(np.all(freq_seqs[2] == np.array([2, 3]), axis=1)))  # [2,3] should be frequent
        
    def test_edge_case_single_sequence(self):
        """Test GSP with a single sequence"""
        sequences = [[1, 2, 3]]
        gsp = GSP(sequences, min_support=1)
        freq_seqs = gsp.find_freq_seq()
        
        # Verify 1-sequences
        self.assertEqual(len(freq_seqs[1]), 3)  # Should have three 1-sequences: 1, 2, 3
        
    def test_seq_to_str(self):
        """Test sequence to string conversion"""
        seq = np.array([1, 2, 3])
        self.assertEqual(GSP.seq_to_str(seq), "{1,2,3}")
        
        seq = np.array([3, 1, 2])
        self.assertEqual(GSP.seq_to_str(seq), "{1,2,3}")  # Should be sorted

if __name__ == '__main__':
    unittest.main()
