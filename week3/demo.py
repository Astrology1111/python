nums = [1,2,3,4,5]
sum_nums = sum(nums)
for i in range(len(nums)):
    nums[i] = nums[i] / sum_nums
print(nums)