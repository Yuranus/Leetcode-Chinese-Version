'''
颠倒给定的 32 位无符号整数的二进制位。

示例:

输入: 43261596
输出: 964176192
解释: 43261596 的二进制表示形式为 00000010100101000001111010011100 ，
     返回 964176192，其二进制表示形式为 00111001011110000010100101000000 。
'''
class Solution:
	# @param n, an integer
	# @return an integer
	def reverseBits(self, n):
		x=bin(n)[2:]
		string=''.join(list(reversed(x)))+(32-len(x))*'0'

		return int(string,2)

s=Solution()
print(s.reverseBits(43261596))
#效率：95.18%#