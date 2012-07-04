function ress = mtimes(a,bb)
if strcmp(a.method,'sparse')
    if (a.adjoint)
        bb = bb(:);
        ress = Tikreg_gridding(a.op,bb,'machine','gpu_float','adjoint');%, ,'single_coil''verbose',1);
    else
        ress = Tikreg_gridding(a.op,bb,'machine','gpu_float','verbose','1');%, 'verbose',1);
    end    
elseif strcmp(a.method,'gridding')
    if (a.adjoint)
        ress = gridding3D_adj(a.op,bb);
    else
        ress = gridding3D_forw(a.op,bb);
    end
end