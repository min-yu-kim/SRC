a = [[(1, 2), (3, 4)], [(5, 6)]]

# 중첩된 리스트를 평탄화
flat_list = [item for sublist in a for item in sublist]

print(flat_list)