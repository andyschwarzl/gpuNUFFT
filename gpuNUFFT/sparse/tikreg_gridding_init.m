%ipk2index
function [idx weight bp_vxidx bp_midx bp_weight] = tikreg_gpuNUFFT_init(ipk,prec)

p = ipk.p;
p = struct(p);
sz = prod(ipk.Jd);%5*5*5

chksum = full(sum(abs(p.arg.G(1,:)))) + p.arg.np + mod(p.arg.odim(1),3123) + size(p.arg.G,1) +prec;

global GPURECO_PLAN;

for k = 1:length(GPURECO_PLAN),
    if GPURECO_PLAN(k).chksum == chksum,
        idx = GPURECO_PLAN(k).idx;
        weight = GPURECO_PLAN(k).weight;
        bp_vxidx = GPURECO_PLAN(k). bp_vxidx;
        bp_midx = GPURECO_PLAN(k).bp_midx;
        bp_weight = GPURECO_PLAN(k).bp_weight;
        return;        
    end;
end;

display('planning');
% sens index generation
idx = zeros(sz,size(p.dim,1));
weight = zeros(2,sz,p.dim(1));

[i j s] = (find(p.arg.G)); %findet alle indizes (zeilen = Trajektoriensample, spalten = Voxelindex) und werte ungleich null
[dummy id] = sort(i); %i...Zeilen, sortiert alle indizes -> speichert werte und indizes in die vektoren
ids = reshape(id,[sz p.dim(1)]); %[125 11685]
    
idx = j(ids); %j...spalten mit kernel voxels
aweight = s(ids); %werte der kernel voxels
weight(1,:,:) = real(aweight);
weight(2,:,:) = imag(aweight);

% backprop index generation
[dummy id] = sort(j); %
bp_vxidx = unique(dummy)-1; % alle von Trajektorie und Kernelgroesse betroffenen Voxel Indizes
boarders =  [1 ; find((dummy(2:end)-dummy(1:end-1))>0) ; length(dummy)];
bp_midx = cell(length(bp_vxidx),1);
bp_weight = cell(length(bp_vxidx),1);
i = uint32(i-1);
if prec == 1,
    s = single(s);
else
    s = double(s);
end;
for k = 1:length(boarders)-1,        
    bp_midx{k} = i(id(boarders(k): boarders(k+1)));
    we = s(id(boarders(k): boarders(k+1)));
    bp_weight{k}(2,:,:) = imag(we);    
    bp_weight{k}(1,:,:) = real(we);   
end;

% save
newplan = length(GPURECO_PLAN)+1;
GPURECO_PLAN(newplan).chksum = chksum;
GPURECO_PLAN(newplan).idx = idx  ;
GPURECO_PLAN(newplan).weight = weight;
GPURECO_PLAN(newplan).bp_vxidx = bp_vxidx;
GPURECO_PLAN(newplan).bp_midx = bp_midx;
GPURECO_PLAN(newplan).bp_weight = bp_weight;