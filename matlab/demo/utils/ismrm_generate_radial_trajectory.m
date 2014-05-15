function [k,w] = ismrm_generate_radial_trajectory(readout_length, projections)
%
%  [k,w] = ismrm_generate_radial_trajectory(readout_length, projections)
%
%  Generates a radial trajctory
%
%  INPUT:
%    - readout_length   : e.g. 256
%    - projections      : e.g. 128
%                                           noise sample BW, etc. into consideration
%
%  OUTPUT:
%    - k  [nsamples, 2]    : k-space coordinates in the range -0.5 to 0.5
%    - w  [nsamples, 1]    : Density compensation weights, scaled such that
%                            sum(w) == numel(w)
%
%   Code made available for the ISMRM 2013 Sunrise Educational Course
% 
%   Michael S. Hansen (michael.hansen@nih.gov)
%

kx = linspace(-.5,.5,readout_length);
k = zeros(projections*numel(kx),2);
w = zeros(size(k,1),1);
for p=1:projections,
    a = p*pi/(projections);
    k((p-1)*numel(kx)+(1:numel(kx)),1) = kx'*cos(a);
    k((p-1)*numel(kx)+(1:numel(kx)),2) = kx'*sin(a);
    w((p-1)*numel(kx)+(1:numel(kx))) = abs(kx);
end

w = w .* (numel(w/sum(w(:))));
