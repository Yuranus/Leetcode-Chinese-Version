'''
给定一个整数 n，返回 n! 结果尾数中零的数量。

示例 1:

输入: 3
输出: 0
解释: 3! = 6, 尾数中没有零。
示例 2:

输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.
说明: 你算法的时间复杂度应为 O(log n) 。
'''
class Solution:
	def trailingZeroes(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n==0:
			return 0

		return int(n/5)+int(self.trailingZeroes(n/5))

s=Solution()
print(s.trailingZeroes(3))
print(s.trailingZeroes(5))
#思路：出现能被5整除的次数#
#效率：未知#


        