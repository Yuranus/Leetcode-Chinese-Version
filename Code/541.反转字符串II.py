# -*- coding: utf-8 -*-
'''
给定一个字符串和一个整数 k，你需要对从字符串开头算起的每个 2k 个字符的前k个字符进行反转。
如果剩余少于 k 个字符，则将剩余的所有全部反转。如果有小于 2k 但大于或等于 k 个字符，则反转前 k 个字符，
并将剩余的字符保持原样。

示例:

输入: s = "abcdefg", k = 2
输出: "bacdfeg"
要求:

该字符串只包含小写的英文字母。
给定字符串的长度和 k 在[1, 10000]范围内。
'''
class Solution:
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        if len(s)==0:
            return ''
        if len(s)<k:
            return ''.join(list(reversed(s)))

        string=''
        length=len(s)//(2*k)
        for i in range(1,length+1):
            temp=s[(i-1)*2*k:i*2*k]
            s_temp=''.join(list(reversed(temp[:k])))
            string+=s_temp+temp[k:]

        final=s[length*2*k:]
        length=len(final)
        if length<k:
            string+=''.join(list(reversed(final)))
            return string
        elif k<=length<2*k:
            temp=final[:k]
            temp=''.join(list(reversed(temp)))
            final=temp+final[k:]
            return string+final

s=Solution()
print(s.reverseStr('abcdefg',2))
#思路：按照题意，分情况讨论即可，注意下标变化。效率：71.38%#


