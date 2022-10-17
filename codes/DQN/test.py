x = [1,2,3,4,5]
index = 4
# reward = sum(map(lambda c, r: 0.99**c * r,
#                     enumerate(x[index-3:index])))
reward = sum(
    [0.99**c * r for c, r in enumerate(x)])
print(reward)