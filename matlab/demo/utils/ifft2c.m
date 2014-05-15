function res = ifft2c(x)
fctr = size(x,1)*size(x,2);
for n=1:size(x,3)
res(:,:,n) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,n))));
end
