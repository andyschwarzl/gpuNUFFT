function res = mtimes(a,b)
% performs the normal nufft
% changed by Clemens Diwoky on 17.12.2009 for 3D
%
% changed by Florian Knoll: 21.02.2011: Adjust dimensions for non-adjoint
% (inverse gridding step)

if a.adjoint
	b = b(:).*a.w(:);
	res = nufft_adj(b, a.st)/sqrt(prod(a.imSize));
	res = reshape(res, a.imSize(1), a.imSize(2), a.imSize(3));
	res = res.*conj(a.phase);
	if a.mode==1
		res = real(res);
	end

else
	b = reshape(b,a.imSize(1),a.imSize(2),a.imSize(3)); % Change Flo
	if a.mode==1
		b = real(b);
	end
	b = b.*a.phase;
	res = nufft(b, a.st)/sqrt(prod(a.imSize));
    % Change Flo: Just a WIP. This has to be commented out because
    % otherwise the dimensions are not consistent. Ask Clemens when he
    % is back why this can work at all if k (*,3) and rawdata (*,1)have
    % different dimensions in the beginning.
	% res = reshape(res,a.dataSize(1),a.dataSize(2));
end



