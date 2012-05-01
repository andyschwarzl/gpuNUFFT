%% 1D fft shift 
N = 6;
X = 0:N-1;
fftshift(X);
Y = fftshift(fftshift(X));
Z = ifftshift(fftshift(X));

hin = mod((X+ceil(N/2)),N);
zruck = mod((hin+floor(N/2)),N);

isequal(zruck,Z)

%% 2D fft shift
N = 4;
NN = N * N -1;
X2D = reshape([0:NN],[4 4])';
fftshift(X2D)

X2D(mod((X2D+ceil(N/2)),N),mod((X2D+ceil(N/2)),N))
