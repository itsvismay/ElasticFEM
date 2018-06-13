A = readDMAT("A.dmat");
B = readDMAT("B.dmat");
[V, D] = eig(A, B);
ev = readDMAT("vecs.dmat");
V(1:10, 1:10)
ev(1:10, 1:10)
diag(D);