class Solution:
    def search(self , nums: List[int], target: int) -> int:
        # write code here
        def find_left() :
            left,right = 0,len(nums)-1
            res = -1
            while left <= right:
                mid = left + (right - left)//2
                if nums[mid] == target:
                    res = mid
                    right = mid -1
                elif nums[mid] < target:
                    left = mid +1
                else:
                    right = mid -1
            return res
        def find_right():
            left,right=0,len(nums)-1
            res = -1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    res = mid
                    left = mid+1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return  res
        left_pos = find_left()
        right_pos = find_right()
        return [left_pos,right_pos]