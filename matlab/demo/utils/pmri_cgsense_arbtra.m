function  us = pmri_cgsense_arbtra(data,FT,c,x0,alpha,tol,maxit)

% [us] = pmri_cgsense_radial(data,FT,c,maxit)
% reconstruct subsampled PMRI data using CG SENSE
%
% INPUT
% data:    3D array of coil images (in k-space)
% FT:      NUFFT operator
% w:       density compensation
% c:       3D array of coil sensitivities
% x0:      prior guess
% alpha:   penalty
% tol:     cg tolerance
% maxit:   maximum number of CG Iterations
%
% OUTPUT
% us:      reconstructed image
%
% Christian Clason (christian.clason@uni-graz.at), February 2010
% Last Change: 15.2.2010
% By: Florian
% =========================================================================

%% set up parameters and operators
[nx,ny] = size(FT'*data(:,1));
nc      = size(c,3);

% sampling operator
F  = @(x) FT*x;
FH = @(x) FT'*x;

% normalize data
dscale = 100/norm(abs(data(:)));
data = data * dscale;

cscale = 0;
for i = 1:nc
    cscale = cscale+abs((c(:,:,i))).^2;
end
cscale = sqrt(cscale);

%% CG iteration

% precompute complex conjugates
cbar = conj(c);

% right hand side: -K^*residual 
y  = zeros(nx,ny);
for i = 1:nc
    y = y + FH(data(:,i)).*cbar(:,:,i);
end

% system matrix: F'^T*F' + alpha I
M  = @(x) applyM(F,FH,c,cbar,x) + alpha*x;

dx = pcg(M,y(:),tol,maxit,[],[],x0(:));

% update iterates
u  = reshape(dx,nx,ny);
% us = cscale.*u./dscale;
us = cscale.*u;

% end main function

%% Derivative evaluation

function y = applyM(F,FH,c,cconj,x)
[nx,ny,nc] = size(c);
dx = reshape(x,nx,ny);

y  = zeros(nx,ny);
for i = 1:nc
    y = y + cconj(:,:,i).*FH(F(c(:,:,i).*dx));
end
y = y(:);
% end function applyM
