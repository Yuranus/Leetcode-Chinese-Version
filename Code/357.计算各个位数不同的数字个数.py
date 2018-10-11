'''
给定一个非负整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10n 。

示例:

输入: 2
输出: 91 
解释: 答案应为除去 11,22,33,44,55,66,77,88,99 外，在 [0,100) 区间内的所有数字。
'''
class Solution:
	def countNumbersWithUniqueDigits(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n==0:
			return 1
		if n==1:
			return 10

		result=10
		f=9
		for i in range(2,n+1):
			f*=(10-i+1)
			result+=f

		return result

s=Solution()
print(s.countNumbersWithUniqueDigits(4))
#思路：数学递推式#
#当n>1时：f(n) = f(n-1)*(10-n+1)，f(1)=9
#当n==1时：返回10，当n为0时，返回0（只有0）#
#效率：94.38%#
        