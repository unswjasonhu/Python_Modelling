from unittest import TestCase
import os

class CustomTestCase(TestCase):
    os.environ["HELLO"] = "WORLD"