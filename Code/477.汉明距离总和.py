'''
两个整数的 汉明距离 指的是这两个数字的二进制数对应位不同的数量。

计算一个数组中，任意两个数之间汉明距离的总和。

示例:

输入: 4, 14, 2

输出: 6

解释: 在二进制表示中，4表示为0100，14表示为1110，2表示为0010。（这样表示是为了体现后四位之间关系）
所以答案为：
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
注意:

数组中元素的范围为从 0到 10^9。
数组的长度不超过 10^4。
'''
class Solution:
	def totalHammingDistance(self, nums):
		"""
		:type nums: List[int]
		:rtype: int
		"""
		count=0
		for i in range(32):
			onecount=0
			for j in range(len(nums)):
				value=nums[j]
				onecount+=((value>>i)&1)
			count+=onecount*(len(nums)-onecount)

		return count

s=Solution()
print(s.totalHammingDistance([4,14,2]))
#思路：最笨的方法超时了毋庸置疑，大神的方法如上：#
#正确的解法应该是针对每一个bit位，统计所有数字在这个bit位上面的1的个数bitCount。#
#那么这一位对结果的贡献就是bitCount*（n-bitCount）#
#效率：70.00%#

