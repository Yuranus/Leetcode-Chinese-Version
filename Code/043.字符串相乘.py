# -*- coding: utf-8 -*-
'''
给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

示例 1:

输入: num1 = "2", num2 = "3"
输出: "6"
示例 2:

输入: num1 = "123", num2 = "456"
输出: "56088"
说明：

num1 和 num2 的长度小于110。
num1 和 num2 只包含数字 0-9。
num1 和 num2 均不以零开头，除非是数字 0 本身。
不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。
'''
class Solution:
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        dict={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
        if num1=='0' or num2=='0':
            return '0'

        n1=0
        n2=0
        for c in num1:
            val=dict[c]
            n1=n1*10+val

        for s in num2:
            val=dict[s]
            n2=n2*10+val

        return str(n1*n2)

s=Solution()
print(s.multiply('2','3'))
print(s.multiply('123','456'))
#效率：85.43%,不能用int直接转，因为直接转float为17位，int为无限位#
#但是可以用字典传入，转为整数#

