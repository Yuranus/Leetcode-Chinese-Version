# -*- coding: utf-8 -*-
'''
给定一个整数，写一个函数来判断它是否是 3 的幂次方。

示例 1:

输入: 27
输出: true
示例 2:

输入: 0
输出: false
示例 3:

输入: 9
输出: true
示例 4:

输入: 45
输出: false
'''
class Solution:
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:
            return False
        while n>1:
            if n%3!=0:
                return False
            n//=3

        return True

s=Solution()
print(s.isPowerOfThree(0))
print(s.isPowerOfThree(27))
print(s.isPowerOfThree(9))
print(s.isPowerOfThree(45))
print(s.isPowerOfThree(-3))
print(s.isPowerOfThree(243))
#效率：41.79%，使用循环结构#


