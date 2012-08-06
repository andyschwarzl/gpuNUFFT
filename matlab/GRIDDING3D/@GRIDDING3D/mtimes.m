function res = mtimes(a,bb)
if strcmp(a.method,'sparse')
    if (a.adjoint)
        bb = bb(:);
        res = Tikreg_gridding(a.op,bb,'machine','gpu_float','adjoint');%, ,'single_coil''verbose',1);
        
        %TODO check
        res = res / sqrt(prod(a.imageDim));
        %END
    else
        res = Tikreg_gridding(a.op,bb,'machine','gpu_float','verbose','1');%, 'verbose',1);
        
        %TODO check
        res = res / sqrt(prod(a.imageDim));
        %END
    end    
elseif strcmp(a.method,'gridding')
    if (a.adjoint)
        res = gridding3D_adj(a.op,bb);
        %TODO check
        res = res / sqrt(prod(a.imageDim));
        %END
    else
        res = gridding3D_forw(a.op,bb);
        %TODO check
        res = res / sqrt(prod(a.imageDim));
        %END
    end
end