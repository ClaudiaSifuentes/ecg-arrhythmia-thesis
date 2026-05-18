import unittest
from src.data.splits import patientwise_split

class TestPatientWiseSplits(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize data for testing
        self.data = [
            {'patient_id': 1, 'ecg_segment': [0.1, 0.2, 0.3]},
            {'patient_id': 1, 'ecg_segment': [0.4, 0.5, 0.6]},
            {'patient_id': 2, 'ecg_segment': [0.7, 0.8, 0.9]},
        ]

    def test_patientwise_split(self):
        # Test the patient-wise split function
        splits = patientwise_split(self.data)
        self.assertEqual(len(splits), 2)  # Expecting 2 patients
        self.assertIn(1, splits)  # Patient 1 should be in splits
        self.assertIn(2, splits)  # Patient 2 should be in splits
        self.assertEqual(len(splits[1]), 2)  # Patient 1 has 2 segments
        self.assertEqual(len(splits[2]), 1)  # Patient 2 has 1 segment

if __name__ == '__main__':
    unittest.main()