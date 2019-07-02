import unittest

import asr_demo


class Asr_demoTestCase(unittest.TestCase):

    def setUp(self):
        self.app = asr_demo.app.test_client()

    def test_index(self):
        rv = self.app.get('/')
        self.assertIn('Welcome to ASRDemo', rv.data.decode())


if __name__ == '__main__':
    unittest.main()
