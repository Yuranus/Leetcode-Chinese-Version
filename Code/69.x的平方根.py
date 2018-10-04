# -*- coding: utf-8 -*-
'''
实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:

输入: 4
输出: 2
示例 2:

输入: 8
输出: 2
说明: 8 的平方根是 2.82842...,
     由于返回类型是整数，小数部分将被舍去。
'''
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==0:
            return 0
        if x==1:
            return 1

        left=1
        right=x
        mid=(left+right)//2

        while left<right:
            if mid**2<x:
                mid+=1
                if mid**2>x:
                    return mid-1
                if mid**2==x:
                    return mid
                left=mid
            elif mid**2>x:
                mid-=1
                if mid**2<=x:
                    return mid
                right=mid
            else:
                return mid
            mid=(left+right)//2

s=Solution()
print(s.mySqrt(100))
#思路：采用二分查找，一般的，会遍历整个n，但是，时间复杂度很大，可以先确立范围再折半，但是判断时候注意一点，#
#可以立马就判断前一个或后一个会更快:O(logx)#

