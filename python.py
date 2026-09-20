mat = [ [ 1 , 2 , 3] , [ 4 , 5 , 6] ,  [7 , 8 , 9] , [ 10 , 11 , 12 ]]

n = len(mat)
m = len(mat[0])

sum = 0
for i in range(n) :
    for j in range(m):
        sum = sum + mat[i][j]
        
print(sum)