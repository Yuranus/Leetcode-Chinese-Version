'''
集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个元素复制了成了集合里面的另外一个元素的值，
导致集合丢失了一个整数并且有一个元素重复。

给定一个数组 nums 代表了集合 S 发生错误后的结果。你的任务是首先寻找到重复出现的整数，再找到丢失的整数，
将它们以数组的形式返回。

示例 1:

输入: nums = [1,2,2,4]
输出: [2,3]
注意:

给定数组的长度范围是 [2, 10000]。
给定的数组是无序的。
'''
class Solution:
	def findErrorNums(self, nums):
		"""
		:type nums: List[int]
		:rtype: List[int]
		"""
		result={}
		final=[]
		for i in range(len(nums)):
			if nums[i] in result:
				result[nums[i]]+=1
				final.append(nums[i])
			else:
				result[nums[i]]=1

		for i in range(1,len(nums)+1):
			if not i in result:
				final.append(i)
				break

		return final

s=Solution()
print(s.findErrorNums([1,2,2,4]))
print(s.findErrorNums([3,2,3,4,6,5]))
#思路：用字典保存数字为键，出现次数为值，次数次数为2就是重复出现的，得到相应的空缺值#
#效率：57.21%#



        