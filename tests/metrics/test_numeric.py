import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.numeric import extract_numeric_answer


class NumericTests(unittest.TestCase):
    def test_integer(self):
        ss = "answer is 123"
        num = extract_numeric_answer(ss)
        self.assertEqual(num, "123")

        ss = "ANSWER: -45.67"
        num = extract_numeric_answer(ss)        
        self.assertEqual("-45", num)

        ss = "\box{0.5}"
        num = extract_numeric_answer(ss)        
        self.assertEqual("0", num)

        ss = "answer is: 1,000"
        num = extract_numeric_answer(ss)        
        self.assertEqual("1000", num)

        ss = "answer is: 1,234.56"
        num = extract_numeric_answer(ss)        
        self.assertEqual("1234", num)

        ss = "answer is 1,000"
        num = extract_numeric_answer(ss)        
        self.assertEqual("1000", num)

        ss = "1e4"
        num = extract_numeric_answer(ss)        
        self.assertEqual("10000", num)        

        ss = "12."
        num = extract_numeric_answer(ss)        
        self.assertEqual("12", num)  

        ss = ".5"
        num = extract_numeric_answer(ss)        
        self.assertEqual("0", num)  

        ss = "-2.3E-1"
        num = extract_numeric_answer(ss)        
        self.assertEqual("0", num)  

    def test_float(self):
        ss = "answer is 123"
        num = extract_numeric_answer(ss, "float")
        self.assertEqual(num, "123.0")

        ss = "ANSWER: -45.67"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("-45.67", num)

        ss = "\box{0.5}"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("0.5", num)

        ss = "answer is: 1,000"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("1000.0", num)

        ss = "answer is: 1,234.56"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("1234.56", num)

        ss = "answer is 1,000"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("1000.0", num)

        ss = "1e4"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("10000.0", num)        

        ss = "12."
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("12.0", num)  

        ss = ".5"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("0.5", num)  

        ss = "-2.3E-1"
        num = extract_numeric_answer(ss, "float")        
        self.assertEqual("-0.23", num)
  
if __name__ == "__main__":
    unittest.main()
