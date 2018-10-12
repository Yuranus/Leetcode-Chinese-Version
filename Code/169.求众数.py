'''
给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在众数。

示例 1:

输入: [3,2,3]
输出: 3
示例 2:

输入: [2,2,1,1,1,2,2]
输出: 2
'''
class Solution:
	def majorityElement(self, nums):
		"""
		:type nums: List[int]
		:rtype: int
		"""
		if len(nums)==1:
			return nums[0]
		result={}
		count=len(nums)//2

		for i in range(len(nums)):
			if nums[i] in result:
				result[nums[i]]+=1
				if result[nums[i]]>count:
					return nums[i]

			else:
				result[nums[i]]=1

s=Solution()
print(s.majorityElement([3,2,3]))
print(s.majorityElement([2,2,1,1,1,2,2]))
#思路：使用字典保存出现的次数，效率：38.90%#
        