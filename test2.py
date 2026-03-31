import unittest

# 需要测试的函数
def multiply(a, b):
    return a * b

# 测试类
class TestMultiply(unittest.TestCase):
    def test_multiply_positive(self):
        self.assertEqual(multiply(3, 4), 12)
    
    def test_multiply_negative(self):
        self.assertEqual(multiply(-1, 5), -5)

if __name__ == '__main__':
    unittest.main()