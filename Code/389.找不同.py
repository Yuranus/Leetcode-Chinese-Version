# -*- coding: utf-8 -*-
'''
给定两个字符串 s 和 t，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。



示例:

输入：
s = "abcd"
t = "abcde"

输出：
e

解释：
'e' 是那个被添加的字母。
'''
class Solution:
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if s==0:
            return t
        s=sorted(list(s))
        t=sorted(list(t))

        for i in range(len(s)):
            if s[i]!=t[i]:
                return t[i]

        return t[-1]

s=Solution()
print(s.findTheDifference('abcd','abcde'))
#思路：先排序，后逐个对比即可，到最后还一样的话，就是t的最后一个#
#效率：65.48%#