# -*- coding: utf-8 -*-
'''
给定一个单词，你需要判断单词的大写使用是否正确。

我们定义，在以下情况时，单词的大写用法是正确的：

全部字母都是大写，比如"USA"。
单词中所有字母都不是大写，比如"leetcode"。
如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
否则，我们定义这个单词没有正确使用大写字母。

示例 1:

输入: "USA"
输出: True
示例 2:

输入: "FlaG"
输出: False
'''
class Solution:
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        if word.isupper() or word.islower():
            return True

        if word[0].isupper() and word[1:].islower():
            return True
        return False

s=Solution()
print(s.detectCapitalUse('USA'))
print(s.detectCapitalUse('FlaG'))
#考察Python的str内部方法的使用，效率：69.18%#


