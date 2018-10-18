# -*- coding: utf-8 -*-
'''
给定一个整数 (32 位有符号整数)，请编写一个函数来判断它是否是 4 的幂次方。

示例 1:

输入: 16
输出: true
示例 2:

输入: 5
输出: false
'''
class Solution:
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num==1:
            return True
        if num==0:
            return False
        while num!=1:
            if num%4!=0:
                return False
            num//=4

        return True

s=Solution()
print(s.isPowerOfFour(16))
print(s.isPowerOfFour(5))
print(s.isPowerOfFour(2))
print(s.isPowerOfFour(-2147483648))
print(s.isPowerOfFour(-2147483647))
