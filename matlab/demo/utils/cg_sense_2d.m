function  u = cg_sense_2d(data,FT,c,mask,alpha,tol,maxit,display,useMulticoil)
% 
% [u] = cg_sense(data,F,Fh,c,mask,alpha,maxit,display)
% reconstruct subsampled PMRI data using CG SENSE [1]
%
% INPUT
% data:    2D array of coil images (in k-space)
% F:       Forward NUFFT operator
% FH:      Adjoint NUFFT operator
% c:       2D array of coil sensitivities
% mask:    region of support for sampling trajectory
% alpha:   penalty for Tikhonov regularization
% tol:     CG tolerance
% maxit:   maximum number of CG iterations
% display: show iteration steps (1) or not (0)
% 
% OUTPUT
% u:       reconstructed 2D image
%
% Original version:
% Christian Clason (christian.clason@uni-graz.at)
% Florian Knoll (florian.knoll@tugraz.at)
% 
% Last Change: Jan 2018
% By: Florian (florian.knoll@nyumc.org)
% 
% [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
% Advances in sensitivity encoding with arbitrary k-space trajectories.
% Magn Reson Med 46: 638-651 (2001)
% 
% =========================================================================
if nargin < 9
  useMulticoil = false;
end
%% set up parameters and operators
[nx,ny,nc] = size(c);

matlabCG = 0;

%% Solve using CG method
% precompute complex conjugates
cbar = conj(c);

% right hand side: -K^*residual 
y  = zeros(nx,ny);
if useMulticoil
    y = FT'* (data .* sqrt(col(mask)));
else
for ii = 1:nc
    y = y + FT' * data(:,ii) .* sqrt(col(mask)) .* cbar(:,:,ii);
end
end

% system matrix: F'^T*F' + alpha I
M  = @(x) applyM(FT,c,cbar,x,useMulticoil) + alpha*x;

%% CG iterations
if matlabCG
    % Use Matlab CG
    x = pcg(M,y(:),tol,maxit);
else
    % Own CG
    x = 0*y(:); r=y(:); p = r; rr = r'*r;
    for it = 1:maxit
        Ap = M(p);
        a = rr/(p'*Ap);
        x = x + a*p;
        rnew = r - a*Ap;
        b = (rnew'*rnew)/rr;
        r=rnew;
        rr = r'*r;
        p = r + b*p;

        if display
            u_it = reshape(x,nx,ny);
            figure(99);
            imshow(abs(u_it),[]); % colorbar;
            title(['Image CG iteration ' num2str(it)]);
            drawnow;
        end
        fprintf('.');
    end
end

% Final reconstructed image
u  = reshape(x,nx,ny);

% Mask k-space with region of support of trajectory: Currently not used
% u =ifft2c(fft2c(u).*mask);

% end main function

%% Derivative evaluation
function y = applyM(FT,c,cconj,x,useMulticoil)
[nx,ny,nc] = size(c);
dx = reshape(x,nx,ny);

y  = zeros(nx,ny);
if useMulticoil
  % full forward/adjoint operator
  % sensitivities are automatically applied
  y = FT'*(FT*dx);
else
  for ii = 1:nc
      y = y + cconj(:,:,ii) .* (FT' * (FT * (c(:,:,ii).*dx)));
  end
end
y = y(:);
% end function applyM
