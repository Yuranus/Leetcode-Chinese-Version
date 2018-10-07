# -*- coding: utf-8 -*-
'''
给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例 1:

输入: "Let's take LeetCode contest"
输出: "s'teL ekat edoCteeL tsetnoc"
注意：在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。
'''
class Solution:
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s)==0 or len(s)==1:
            return s
        words=s.split()
        string=''
        for word in words:
            string+=''.join(list(reversed(list(word))))+' '

        return string[:-1]

s=Solution()
print(s.reverseWords("Let's take LeetCode contest"))
#思路：按照空格split再对每个反转即可，效率：38.71%#
