import unittest

from adpilot.backends.video import WanImageToVideoBackend


class BackendReleaseTests(unittest.TestCase):
    def test_release_discards_loaded_pipeline_and_clears_cuda_cache(self):
        calls = []

        class TorchStub:
            class cuda:
                @staticmethod
                def empty_cache():
                    calls.append("empty_cache")

        backend = WanImageToVideoBackend(model_id="unused", device="cuda")
        backend._pipe = object()
        backend._torch = TorchStub()

        backend.release()

        self.assertIsNone(backend._pipe)
        self.assertEqual(calls, ["empty_cache"])


if __name__ == "__main__":
    unittest.main()
