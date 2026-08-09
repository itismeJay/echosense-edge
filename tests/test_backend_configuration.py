import unittest

from config import backend_base_url_from_environment


class BackendConfigurationTests(unittest.TestCase):
    def test_local_integration_backend_loads_from_environment(self):
        self.assertEqual(
            backend_base_url_from_environment(
                {
                    "ECHOSENSE_BACKEND_BASE_URL": (
                        "  http://192.168.1.93:8000/  "
                    )
                }
            ),
            "http://192.168.1.93:8000",
        )

    def test_production_backend_remains_selectable(self):
        production_url = "https://echosense-backend-75h3.onrender.com"
        self.assertEqual(
            backend_base_url_from_environment(
                {"ECHOSENSE_BACKEND_BASE_URL": production_url}
            ),
            production_url,
        )

    def test_missing_backend_has_no_render_fallback(self):
        self.assertEqual(backend_base_url_from_environment({}), "")


if __name__ == "__main__":
    unittest.main()
