'''
给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

示例 1:

输入: "abab"

输出: True

解释: 可由子字符串 "ab" 重复两次构成。
示例 2:

输入: "aba"

输出: False
示例 3:

输入: "abcabcabcabc"

输出: True

解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
'''
class Solution:
	def repeatedSubstringPattern(self, s):
		"""
		:type s: str
		:rtype: bool
		"""
		if len(s)==0 or len(s)==1:
			return False

		temp=''
		for i in range(len(s)-1):
			temp+=s[i]
			length=len(temp)
			if len(s)%length==0:
				res=len(s)//length
				if temp*res==s:
					return True

		return False

s=Solution()
print(s.repeatedSubstringPattern('abab'))
print(s.repeatedSubstringPattern('aba'))
print(s.repeatedSubstringPattern('abcabcabcabcabc'))        
#思路：将原有的每个字符加上，判断其整个是不是这些加上字符的整数倍即可#
#效率：28.00%#