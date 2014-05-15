function  [data, sp, smaps, Rn, gt_image] = generate_data(acc_factor,refL,coilset,noise_scale)

load im1.mat
load smaps_phantom.mat
load noise_covariances.mat

L_normal_8 = chol(Rn_normal_8,'lower');
L_broken_8 = chol(Rn_broken_8,'lower');
ncoils = size(smaps,3);

%Some settings
noise_level = noise_scale*max(im1(:));
noise_white = noise_level*complex(randn(size(im1,1),size(im1,2),size(smaps,3)),randn(size(im1,1),size(im1,2),size(smaps,3)));

% generate correlated noise
switch coilset
    case 'normal'
noise_color = reshape(permute(L_normal_8 * permute(reshape(noise_white, numel(noise_white)/ncoils,ncoils),[2 1]),[2 1]),size(noise_white));
    Rn = Rn_normal_8;    
    case 'broken'
noise_color = reshape(permute(L_broken_8 * permute(reshape(noise_white, numel(noise_white)/ncoils,ncoils),[2 1]),[2 1]),size(noise_white));
    Rn = Rn_broken_8;    
    otherwise
        error('no valid coilset')
end

[data, sp] = ismrm_sample_data(im1, smaps, acc_factor, refL);
data = data + noise_color .* repmat(sp > 0,[1 1 ncoils]);
gt_image = im1;
end


