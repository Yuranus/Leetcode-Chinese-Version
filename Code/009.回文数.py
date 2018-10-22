# -*- coding: utf-8 -*-
'''
判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

示例 1:

输入: 121
输出: true
示例 2:

输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
示例 3:

输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。
进阶:

你能不将整数转为字符串来解决这个问题吗？
'''
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if 0<=x<10:
            return True
        if x<0:
            return False
        res=[]
        while x:
            res.append(x%10)
            x//=10

        left=0
        right=len(res)-1
        while left<=right:
            if res[left]==res[right]:
                left+=1
                right-=1
            else:
                return False

        return True

s=Solution()
print(s.isPalindrome(121))
print(s.isPalindrome(-121))
print(s.isPalindrome(10))
#思路：不用字符串的形式就是要遍历，时间会很久，方法就是得到x每位数字，分别从左往右和从右往左遍历#
#相同时继续，不相同为False#
#效率：12.64%#

