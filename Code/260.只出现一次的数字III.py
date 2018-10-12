'''
给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

示例 :

输入: [1,2,1,3,2,5]
输出: [3,5]
注意：

结果输出的顺序并不重要，对于上面的例子， [5, 3] 也是正确答案。
你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？
'''
class Solution:
	def singleNumber(self, nums):
		"""
		:type nums: List[int]
		:rtype: List[int]
		"""
		result=nums[0]
		for i in range(1,len(nums)):
			result^=nums[i]

		index=0
		while ((result>>index)&1)==0:
			index+=1

		res1=0
		res2=0
		for i in range(len(nums)):
			if (nums[i]>>index)&1:
				res1^=nums[i]

			else:
				res2^=nums[i]

		return [res1,res2]

s=Solution()
print(s.singleNumber([1,2,1,3,2,5]))
#思路：利用找单独出现一次数字的解题思路，将所有数字异或，这样得到的就是两个出现一次数的异或结果。
#然后从右往左找到异或结果数位第一位为1的位置，然后关于此位是否为1将数组分为两部分，
#这样两个数字分别分到了两部分，并且每部分里的其他数都成对出现。再对两部分分别异或运算，得到的结果便是两个出现一次的数。#
#效率：38.71%#





