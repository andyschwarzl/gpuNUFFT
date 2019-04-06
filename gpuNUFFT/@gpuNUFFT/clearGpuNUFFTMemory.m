function [] = clearGpuNUFFTMemory(operator)
% Clears operator from memory
% Frees GPU memory usage by clearing used mex files
disp('Clearing loaded mex files');
clear @gpuNUFFT/private/mex_gpuNUFFT_precomp_f;
clear @gpuNUFFT/private/mex_gpuNUFFT_adj_atomic_f;
clear @gpuNUFFT/private/mex_gpuNUFFT_forw_atomic_f;

end

