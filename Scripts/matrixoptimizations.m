c = [[5 4];[-4 5]];
l = kron([1;1], eye(2));
C = l*c*l';
K = kron(eye(2), c);