# -*- coding: utf-8 -*-
'''
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
示例 2:

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
'''
class Solution:
    def binary(self,left,right,nums,target):
        mid=(left+right)//2
        while left<=right:
            if nums[mid]<target:
                left=mid+1
            elif nums[mid]>target:
                right=mid-1
            else:
                return mid

            mid=(left+right)//2

        return -1

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums)==0:
            return -1

        if len(nums)==1:
            if target==nums[0]:
                return 0
            return -1

        if len(nums)==2:
            if target in nums:
                return nums.index(target)
            return -1

        index=-1
        for i in range(1,len(nums)):
            if nums[i]<nums[i-1]:
                index=i
                break

        if index==-1:
            return self.binary(0,len(nums)-1,nums,target)

        if nums[index]==target:
            return index

        res1=self.binary(0,index-1,nums[:index],target)
        if res1!=-1:
            return res1

        res2=self.binary(0,len(nums[index:])-1,nums[index:],target)
        if res2!=-1:
            return res2+index

        return -1

s=Solution()
print(s.search([4,5,6,7,0,1,2],0))
print(s.search([4,5,6,7,0,1,2],3))
print(s.search([1,3,5],1))
print(s.search([1,3,5],5))
print(s.search([5,1,3],3))
#思路：首先找出旋转的位置，然后对每个分数组进行二分查找，注意第二个数组的下标#
#效率：52.56%#







