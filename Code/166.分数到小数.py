'''
给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以字符串形式返回小数。

如果小数部分为循环小数，则将循环的部分括在括号内。

示例 1:

输入: numerator = 1, denominator = 2
输出: "0.5"
示例 2:

输入: numerator = 2, denominator = 1
输出: "2"
示例 3:

输入: numerator = 2, denominator = 3
输出: "0.(6)"
'''
class Solution:
	def fractionToDecimal(self, numerator, denominator):
		"""
		:type numerator: int
		:type denominator: int
		:rtype: str
		"""
		if numerator==0:
			return '0'
		if denominator==0:
			return ''

		res=''

		n=numerator
		d=denominator
		if (n<0 and d>0) or (n>0 and d<0):
			res='-'

		if n<0:
			n=-n
		if  d<0:
			d=-d

		res+=str(n//d)
		n=n%d
		if n==0:
			return res
		n*=10
		result={}
		res+='.'

		while n!=0:
			if n in result:
				begin=result.get(n)
				part1=res[:begin]
				part2=res[begin:]
				res=part1+'('+part2+')'
				return res

			result[n]=len(res)
			temp=n//d
			res+=str(temp)
			n=(n%d)*10

		return res

s=Solution()
print(s.fractionToDecimal(1,2))
print(s.fractionToDecimal(2,1))
print(s.fractionToDecimal(2,3))
#思路：正负数判断，然后整除判断，然后小数再*10和denominator相除#
#将小数放入dict，如果之前出现了，就说明循环开始。直接return即可#
#否则就一直相除直到n=0#
#效率：97.78%#