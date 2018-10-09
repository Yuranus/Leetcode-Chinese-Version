'''
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

案例:

s = "leetcode"
返回 0.

s = "loveleetcode",
返回 2.
 

注意事项：您可以假定该字符串只包含小写字母。
'''
class Solution:
	def firstUniqChar(self, s):
		"""
		:type s: str
		:rtype: int
		"""
		if len(s)==0:
			return -1
		if len(s)==1:
			return 0

		for i in range(len(s)):
			if not s[i] in s[:i]+s[i+1:]:
				return i

		return -1

s=Solution()
print(s.firstUniqChar('leetcode'))
print(s.firstUniqChar('loveleetcode'))
#思路：检测在不在原字符串里面即可#
#效率：6.83%（太低啦）#