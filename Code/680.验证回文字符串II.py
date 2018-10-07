# -*- coding: utf-8 -*-
'''
给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。

示例 1:

输入: "aba"
输出: True
示例 2:

输入: "abca"
输出: True
解释: 你可以删除c字符。
'''
class Solution:
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s)==0 or len(s)==1:
            return True

        if s==''.join(list(reversed(s))):
            return True

        s=list(s)
        left=0
        right=len(s)-1

        while left<=right:
            if s[left]==s[right]:
                left+=1
                right-=1
            else:
                break

        s1=s[:left]+s[left+1:]
        if s1==list(reversed(s1)):
            return True
        s2=s[:right]+s[right+1:]
        if s2==list(reversed(s2)):
            return True

        return False

s=Solution()
print(s.validPalindrome('aba'))
print(s.validPalindrome('abca'))
print(s.validPalindrome('abcda'))
print(s.validPalindrome('cbbcc'))
#思路：回文串是逆置和原始相同的串，两个标识位，同时移动，如果遇到不一样的一定又一个要删除，删除过后是回文#
#就return True，否则return False#
#效率：61.54%#


