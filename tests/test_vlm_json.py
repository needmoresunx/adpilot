import unittest

from adpilot.identity.vlm import parse_json_response


class VlmJsonTests(unittest.TestCase):
    def test_accepts_json_inside_markdown(self):
        self.assertEqual(parse_json_response("Here:\n```json\n{\"name\": \"Miss Dior\"}\n```"), {"name": "Miss Dior"})

    def test_repairs_only_trailing_commas(self):
        self.assertEqual(parse_json_response('{"shots": [{"goal": "hook",},],}'), {"shots": [{"goal": "hook"}]})

    def test_invalid_json_reports_location_and_excerpt(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON at line"):
            parse_json_response('{"name": "unfinished" "extra": true}')


if __name__ == "__main__":
    unittest.main()
