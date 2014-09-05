function  u = cg_sense_3d(data,FT,c,mask,alpha,tol,maxit,display,slice,useMulticoil)
% 
% [u] = cg_sense(data,F,Fh,c,mask,alpha,maxit,display)
% reconstruct subsampled PMRI data using CG SENSE [1]
%
% INPUT
% data:    3D array of coil images (in k-space)
% F:       Forward NUFFT operator
% FH:      Adjoint NUFFT operator
% c:       3D array of coil sensitivities
% mask:    region of support for sampling trajectory
% alpha:   penalty for Tikhonov regularization
% tol:     CG tolerance
% maxit:   maximum number of CG iterations
% display: show iteration steps (1) or not (0)
% slice:   slice for display
% 
% OUTPUT
% u:       reconstructed 3D image volume
%
% Last Change: 23.11.2012
% By: Florian (florian.knoll@tugraz.at)
% 
% [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
% Advances in sensitivity encoding with arbitrary k-space trajectories.
% Magn Reson Med 46: 638-651 (2001)
% 
% =========================================================================
if nargin < 10
  useMulticoil = false;
end
%% set up parameters and operators
%[nx,ny,nz] = size(FH(data(:,:,1,1)));
[nx,ny,nz,nc] = size(c);

matlabCG = 1;

%% Solve using CG method
% precompute complex conjugates
cbar = conj(c);

% right hand side: -K^*residual 
y  = zeros(nx,ny,nz);
if useMulticoil
    y = FT'*data;
else
for ii = 1:nc
    y = y + FT*(data(:,:,ii)).*cbar(:,:,:,ii);
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
            u_it = reshape(x,nx,ny,nz);
            u_it = u_it(:,:,slice);
            figure(99);
            imshow(abs(u_it),[]); % colorbar;
            title(['Image CG iteration ' num2str(it)]);
            drawnow;
        end
        fprintf('.');
    end
end

% Final reconstructed image
u  = reshape(x,nx,ny,nz);

% Mask k-space with region of support of trajectory: Currently not used
% u =ifft2c(fft2c(u).*mask);

% end main function

%% Derivative evaluation
function y = applyM(FT,c,cconj,x,useMulticoil)
[nx,ny,nz,nc] = size(c);
dx = reshape(x,nx,ny,nz);

y  = zeros(nx,ny,nz);
if useMulticoil
  % full forward/adjoint operator
  % sensitivities are automatically applied
  y = FT'*(FT*dx);
else
  for ii = 1:nc
      y = y + cconj(:,:,:,ii).*FH(F(c(:,:,:,ii).*dx));
  end
end
y = y(:);
% end function applyM
