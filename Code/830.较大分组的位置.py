'''
在一个由小写字母构成的字符串 S 中，包含由一些连续的相同字符所构成的分组。

例如，在字符串 S = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。

我们称所有包含大于或等于三个连续字符的分组为较大分组。找到每一个较大分组的起始和终止位置。

最终结果按照字典顺序输出。

示例 1:

输入: "abbxxxxzzy"
输出: [[3,6]]
解释: "xxxx" 是一个起始于 3 且终止于 6 的较大分组。
示例 2:

输入: "abc"
输出: []
解释: "a","b" 和 "c" 均不是符合要求的较大分组。
示例 3:

输入: "abcdddeeeeaabbbcd"
输出: [[3,5],[6,9],[12,14]]
说明:  1 <= S.length <= 1000
'''
class Solution:
	def largeGroupPositions(self, S):
		"""
		:type S: str
		:rtype: List[List[int]]
		"""
		if len(S)==0:
			return []

		start=0
		final=[]
		count=1
		for i in range(1,len(S)):
			if S[i]==S[i-1]:
				count+=1
			else:
				if count>=3:
					final.append([start,start+count-1])

				count=1
				start=i

		if count>=3:
			final.append([start,start+count-1])
			
		return final

s=Solution()
print(s.largeGroupPositions('abbxxxxzzy'))
print(s.largeGroupPositions('abc'))
print(s.largeGroupPositions('abcdddeeeeaabbbcd'))
print(s.largeGroupPositions('aaa'))
#思路：思路简单，遍历字符串，和前面不一样的就计数从1开始，然后决定start从那个位置开始，count超过3的就append进final中#
#效率：40.06%#

        