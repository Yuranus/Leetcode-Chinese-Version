# -*- coding: utf-8 -*-
'''
给定一个仅包含大小写字母和空格 ' ' 的字符串，返回其最后一个单词的长度。

如果不存在最后一个单词，请返回 0 。

说明：一个单词是指由字母组成，但不包含任何空格的字符串。

示例:

输入: "Hello World"
输出: 5
'''
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s)==0 or s.isspace():
            return 0

        s=s.split()

        return len(s[-1])

s=Solution()
print(s.lengthOfLastWord('Hello World'))
print(s.lengthOfLastWord('a '))



