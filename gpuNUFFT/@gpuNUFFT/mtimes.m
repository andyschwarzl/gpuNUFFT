function res = mtimes(a,bb)
if (a.adjoint)
  res = gpuNUFFT_adj(a.op,bb);
else
  res = gpuNUFFT_forw(a.op,bb);
end