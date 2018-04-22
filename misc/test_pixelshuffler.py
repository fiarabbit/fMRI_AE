from numpy import *

B = 3
C = 2
X = 8
Y = 8
Z = 8
r = 2

m = zeros(B * C * X * Y * Z).reshape((B, C, X, Y, Z))
for b in range(B):
    for c in range(C):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    m[b, c, x, y, z] = ((x % 2 == 0) + 2 * (y % 2 == 0) + 4 * (z % 2 == 0)) * (10 ** c)


# Reorg layer
_ = reshape(m, (B, C, int(X / r), r, int(Y / r), r, int(Z / r), r)) # -> b, c, x, r, y, r, z, r
_ = transpose(_, (0, 3, 5, 7, 1, 2, 4, 6)) # -> b, r, r, r, c, x, y, z
_ = reshape(_, (B, C * r ** 3, int(X / r), int(Y / r), int(Z / r))) # -> b, c, x, y, z
# _ = transpose(_, (1, 0, 2, 3)) # -> b, c, x, y

_ = transpose(m, (1, 0, 2, 3)) # -> c, b, x, y
_ = reshape(_, (C, B, int(X/r), r, int(Y/r), r))
_ = transpose(_, (0, 3, 5, 1, 2, 4))
_ = reshape(_, (C, B*r**2, int(X/r), int(Y/r)))
_ = transpose(_, (1, 0, 2, 3))


B = 12
X = 4
Y = 4
r = 2

m = zeros(B * X * Y).reshape((B, X, Y))
for b in range(B):
    for x in range(X):
        for y in range(Y):
            if (b < B/ r**2):
                m[b, x, y] = 1
            elif (b <  2 *B / r**2):
                m[b, x, y] = 2
            elif (b < 3 * B / r ** 2):
                m[b, x, y] = 3
            else:
                m[b, x, y] = 4

_ = reshape(m, (r, r, int(B/r**2), X, Y))
_ = transpose(_, (2, 3, 0, 4, 1))
_ = reshape(_, (int(B/r**2), X*r, Y*r))
