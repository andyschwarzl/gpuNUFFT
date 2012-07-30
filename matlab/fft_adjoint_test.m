%%
N = [8 1];
xtest =  randn(N);
Ktest = randn(N);
ktest = (1/sqrt(prod(N)))*ifft(Ktest);
Xtest = (1/sqrt(prod(N)))*fft(xtest);
xtest(:)' * ktest(:) - Xtest(:)' * Ktest(:)

%%
Ksim = fft(xtest);
xsim = ifft(Ksim);
xtest - xsim