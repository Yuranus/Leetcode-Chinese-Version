# -*- coding: utf-8 -*-
'''
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

示例 1:

输入: "A man, a plan, a canal: Panama"
输出: true
示例 2:

输入: "race a car"
输出: false
'''
class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s)==0 or len(s)==1:
            return True
        s=s.lower()

        left=0
        right=len(s)-1

        while left<=right:
            if s[left]==s[right]:
                left+=1
                right-=1
            elif not s[left].isalnum():
                left+=1
            elif not s[right].isalnum():
                right-=1
            else:
                return False

        return True

s=Solution()
print(s.isPalindrome('race a car'))
#思路：本题思路很简单，left和right分别遍历即可。#
#效率：91.91%#
