import unittest
from utils.utils import prepare_playpen_results


class TestPreparePlaypenResults(unittest.TestCase):
    def test_with_valid_harness_results(self):
        task_name = "mmlu"
        model_name = "gemma"
        harness_results = {
            "results": {
                "task1": {
                    "accuracy": 0.85,
                    "f1_score": 0.8,
                    "acc_stderr": 0.05,
                    "precision": 0.75,
                }
            }
        }

        # Expected result should only consider 'accuracy' or 'f1_score' but not 'acc_stderr'
        expected_output = {
            "model_name": "modelA",
            "task": "task1",
            "aggregated_results": {"accuracy": 0.85},
            "subtask_results": {}
        }

        output = prepare_playpen_results(task_name, model_name, harness_results)
        self.assertEqual(output, expected_output)

    def test_no_harness_results_provided(self):
        task_name = "task1"
        model_name = "modelA"

        with self.assertRaises(Exception) as context:
            prepare_playpen_results(task_name, model_name, None)

        self.assertIn("Other options besides dealing with simplifying results from harness are not yet implemented.",
                      str(context.exception))

    def test_multiple_score_keys(self):
        task_name = "task1"
        model_name = "modelB"
        harness_results = {
            "results": {
                "task1": {
                    "accuracy": 0.9,
                    "f1_score": 0.88,
                    "f1_stderr": 0.03,
                    "recall": 0.85,
                }
            }
        }

        # Expected output with the first matching score key
        expected_output = {
            "model_name": "modelB",
            "task": "task1",
            "aggregated_results": {"accuracy": 0.9},
            "subtask_results": {}
        }

        output = prepare_playpen_results(task_name, model_name, harness_results)
        self.assertEqual(output, expected_output)

    def test_no_matching_score_key(self):
        task_name = "task1"
        model_name = "modelC"
        harness_results = {
            "results": {
                "task1": {
                    "precision": 0.7,
                    "recall": 0.8,
                    "other_metric": 0.65,
                }
            }
        }

        # Should result in an empty aggregated_results as no 'acc' or 'f1' keys match
        expected_output = {
            "model_name": "modelC",
            "task": "task1",
            "aggregated_results": {},
            "subtask_results": {}
        }

        output = prepare_playpen_results(task_name, model_name, harness_results)
        self.assertEqual(output, expected_output)


# Run the tests
if __name__ == '__main__':
    unittest.main()
