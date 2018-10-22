# -*- coding: utf-8 -*-
'''
对于一个 正整数，如果它和除了它自身以外的所有正因子之和相等，我们称它为“完美数”。

给定一个 正整数 n， 如果他是完美数，返回 True，否则返回 False



示例：

输入: 28
输出: True
解释: 28 = 1 + 2 + 4 + 7 + 14


注意:

输入的数字 n 不会超过 100,000,000. (1e8)
'''
import math
class Solution:
    def checkPerfectNumber(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num==2 or num<=0 or num==1:
            return False

        res=1
        for i in range(2,int(math.sqrt(num))+1):
            if not num%i:
                res+=i
                res+=num//i

        if res==num:
            return True
        return False

s=Solution()
print(s.checkPerfectNumber(28))
print(s.checkPerfectNumber(3))
print(s.checkPerfectNumber(2976221))
print(s.checkPerfectNumber(13466917))
#思路：一个数的所有一半因子在sqrt(num)之前#
#效率：74.54%#