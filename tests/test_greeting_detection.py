import unittest


try:
    from learning_orchestrator import _contains_greeting
except Exception as exc:  # pragma: no cover - safeguard when deps are missing
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"learning_orchestrator import failed: {_IMPORT_ERROR}")
class GreetingDetectionTests(unittest.TestCase):
    def test_standalone_greeting(self):
        self.assertTrue(_contains_greeting("hey"))

    def test_capitalized_greeting(self):
        self.assertTrue(_contains_greeting("Hi there!"))

    def test_substring_not_greeting(self):
        self.assertFalse(_contains_greeting("they are here"))

    def test_suffix_substring_not_greeting(self):
        self.assertFalse(_contains_greeting("ohey"))

    def test_empty_string(self):
        self.assertFalse(_contains_greeting(""))

    def test_multi_word_greeting(self):
        self.assertTrue(_contains_greeting("good morning everyone"))


if __name__ == "__main__":
    unittest.main()
