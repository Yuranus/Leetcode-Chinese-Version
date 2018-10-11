'''
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

示例 1:

输入: [1,3,5,6], 5
输出: 2
示例 2:

输入: [1,3,5,6], 2
输出: 1
示例 3:

输入: [1,3,5,6], 7
输出: 4
示例 4:

输入: [1,3,5,6], 0
输出: 0
'''
class Solution:
	def searchInsert(self, nums, target):
		"""
		:type nums: List[int]
		:type target: int
		:rtype: int
		"""
		left=0
		right=len(nums)-1
		mid=(left+right)//2
		if target>nums[-1]:
			return len(nums)
		if target<nums[0]:
			return 0

		while left<=right:
			if nums[mid]<target:
				left=mid+1
			elif nums[mid]>target:
				right=mid-1
			else:
				return mid

			mid=(left+right)//2

		return mid+1

s=Solution()
print(s.searchInsert([1,3,5,6],5))
print(s.searchInsert([1,3,5,6],2))
print(s.searchInsert([1,3,5,6],7))
print(s.searchInsert([1,3,5,6],0))
print(s.searchInsert([1,3],2))
#思路：最简单的二分查找，对于插入位置，要注意mid的取值#
#效率：75.84%#

        