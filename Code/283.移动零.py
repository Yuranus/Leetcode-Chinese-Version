# -*- coding: utf-8 -*-
'''
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:

输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:

必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
'''
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i=0
        length=len(nums)
        count=0
        while i<(length-count):
            if nums[i]==0:
                nums.append(nums.pop(i))
                count+=1
            else:
                i+=1

        return nums

s=Solution()
print(s.moveZeroes([0,1,0,3,12]))
print(s.moveZeroes([0,0,1]))
#思路：此题思路易懂，首先我们设置指针i指向当前位置，可是nums是动态变化的，
# 所以，要判断结束的方法是非零数和零的数之和为数组长度即可退出#
#效率：69.91%#

