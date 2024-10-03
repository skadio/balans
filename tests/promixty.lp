\ SCIP STATISTICS
\   Problem name     : model
\   Variables        : 8 (2 binary, 3 integer, 0 implicit integer, 3 continuous)
\   Constraints      : 3
Minimize
 Obj: -1 x1 -1 x2 +1000 x8 +2
Subject to
 c1: +1 x3 +1 x4 +1 x5 +1 x6 +1 x7 = +60
 c2: +1 x1 +1 x2 = +1
 c3: -1 x3 -1 x4 -1 x6 -1 x8 <= -39.8
Bounds
 0 <= x1 <= 1
 0 <= x2 <= 1
Binaries
 x1 x2
Generals
 x3 x4 x5
End
