function  res = NUFFT3D(k,w,phase,shift,imSize, mode)
% 
%	Interface to Jeffery Fessler's NUFFT Tolbox, 3D NUFFT
% 
%   res = NUFFT3D(k,w,phase,shift,imSize,mode)
%
%	Inputs:
%		k - normalized kspace coordinates (between -0.5 to 0.5)
%`		w - density compensation (w=1, means no compensation)
%		phase - phase of the image for phase correction
%		shift - shift the image center
%		imSize - the image size
%		mode - 1 - contrain image to be real, 2 - complex image
%
%	Outputs:
%		res - the NUFFT operator
%
%   Extension to 3D by Florian Knoll, based on  Miki Lustig's 2D interface
%
%   Note: If you want to use table based interpolation, you need the
%   corresponding compiled mexfile for your system. If you have problems
%   with this, use the sparse matrix version instead (line 36 instead of
%   line 37), but this needs a lot more memory.


if exist('nufft') <2
	error('must have Jeffery Fessler''s NUFFT code in the path');
end

    om = k*2*pi;
    clear k;
    Nd = [imSize];
    Jd = [6,6,6];
    Kd = [Nd*2];
    n_shift = Nd/2 + shift; 
    % n_shift = [0,0,0] + shift;
    % res.st = nufft_init(om, Nd, Jd, Kd, n_shift);
    res.st = nufft_init(om, Nd, Jd, Kd, n_shift, 'table', 2^12, 'minmax:kb'); 

    res.phase = phase;
    res.adjoint = 0;
    res.imSize = imSize;
    res.dataSize = size(om);
    res.w = w;
    res.mode = mode;
    res = class(res,'NUFFT3D');

