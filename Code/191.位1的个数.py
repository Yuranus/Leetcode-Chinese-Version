# -*- coding: utf-8 -*-
'''
编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。

示例 :

输入: 11
输出: 3
解释: 整数 11 的二进制表示为 00000000000000000000000000001011


示例 2:

输入: 128
输出: 1
解释: 整数 128 的二进制表示为 00000000000000000000000010000000
'''
import re

class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        string=bin(n)[2:]
        reg=re.compile('(?=1)')

        return len(reg.findall(string))

s=Solution()
print(s.hammingWeight(128))
print(s.hammingWeight(11))
#思路：求出的二进制使用正则表达式会比str.count快#
#效率：93.39%#


