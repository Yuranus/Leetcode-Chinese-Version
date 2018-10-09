'''
给定一个二进制数组， 计算其中最大连续1的个数。

示例 1:

输入: [1,1,0,1,1,1]
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.
注意：

输入的数组只包含 0 和1。
输入数组的长度是正整数，且不超过 10,000。
'''
class Solution:
	def findMaxConsecutiveOnes(self, nums):
		"""
		:type nums: List[int]
		:rtype: int
		"""
		if len(nums)==0:
			return 0

		nums=''.join(list(map(str,nums)))
		nums=nums.split('0')

		nums.sort(key=lambda x : len(x))

		return len(nums[-1])

s=Solution()
print(s.findMaxConsecutiveOnes([1,1,0,1,1,1]))
#思路：将原数组按照0分开成不同的数组，求最长的那个数组长度即可#
#效率：59.13%#
        