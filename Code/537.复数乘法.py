'''
给定两个表示复数的字符串。

返回表示它们乘积的字符串。注意，根据定义 i2 = -1 。

示例 1:

输入: "1+1i", "1+1i"
输出: "0+2i"
解释: (1 + i) * (1 + i) = 1 + i2 + 2 * i = 2i ，你需要将它转换为 0+2i 的形式。
示例 2:

输入: "1+-1i", "1+-1i"
输出: "0+-2i"
解释: (1 - i) * (1 - i) = 1 + i2 - 2 * i = -2i ，你需要将它转换为 0+-2i 的形式。 
注意:

输入字符串不包含额外的空格。
输入字符串将以 a+bi 的形式给出，其中整数 a 和 b 的范围均在 [-100, 100] 之间。输出也应当符合这种形式。
'''
class Solution:
	def complexNumberMultiply(self, a, b):
		"""
		:type a: str
		:type b: str
		:rtype: str
		"""
		a=a.split('+')
		b=b.split('+')
		res1=int(a[0])*int(b[0])
		res2=int(a[0])*int(b[1][:-1])
		res3=int(a[1][:-1])*int(b[0])
		res4=int(a[1][:-1])*int(b[1][:-1])

		res23=res2+res3

		res4*=(-1)

		res14=res1+res4

		string=str(res14)+'+'+str(res23)+'i'

		return string

s=Solution()
print(s.complexNumberMultiply('1+1i','1+1i'))
print(s.complexNumberMultiply('1+-1i','1+-1i'))
#思路：二项展开式求解，注意i^2=-1,效率：99.27%#
