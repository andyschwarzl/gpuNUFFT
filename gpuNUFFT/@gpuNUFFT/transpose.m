function res = transpose(a)
a.adjoint = xor(a.adjoint,1);
res = a;

