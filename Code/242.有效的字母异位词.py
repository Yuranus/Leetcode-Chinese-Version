# -*- coding: utf-8 -*-
'''
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的一个字母异位词。

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
说明:
你可以假设字符串只包含小写字母。

进阶:
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？
'''
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if not len(s)==len(t):
            return False
        resA={}
        for i in range(len(s)):
            if s[i] in resA:
                resA[s[i]]+=1
            else:
                resA[s[i]]=1

        resB={}
        for i in range(len(t)):
            if t[i] in resB:
                resB[t[i]]+=1
            else:
                resB[t[i]]=1

        for key,value in resA.items():
            if not key in resB or (key in resB and value!=resB[key]):
                return False
        return True

s=Solution()
print(s.isAnagram('anagram','nagaram'))
print(s.isAnagram('rat','car'))
#思路：一开始用排序的方式，但是Python的sorted时间为O(nlogn)，会很慢#
#转而采用遍历，字典的方式，因为字典的查找为O(1)，遍历为O(n),效率：49.61%#
