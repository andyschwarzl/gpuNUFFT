function [k_traj,dataRadial,dens] = create_phantom(N, radialTraj, nDim,R, n3D)  

if nargin < 5
    n3D = 1;
end
if nargin < 4
    R = 1;
end

javaaddpath ../../../../STBB/STBBFolder/JarFile/STBB.jar
import STBB.*
sl2d = SheppLogan2D(); 
sl3d = SheppLogan3D(); 

%% build kspace trajectory
% phantom kspace has to be scaled
% from -32 - 32
%
pR = 32;

% res of image
% del_x = FOV/N
%
% res of kspace
% FOV = -1 to 1
% del_kx = 1/FOV = 0.5.
% |k_max| is given by del_kx * N/2
pR = N / 4;

if (radialTraj)
    rho=linspace(-pR,pR,N)';
    numSpokes = N/2;
    nRO = N/2;

    % Trajectory
    theta=linspace(0,pi-pi/numSpokes,numSpokes);
    phi=linspace(-pi,pi-pi/nRO,nRO)';

    if (nDim == 2)
        kx = rho*cos(theta);
        ky = rho*sin(theta);
    
        k_traj = [kx(:) ky(:)];
        dens = col(repmat(abs(rho),[1 numSpokes]));
    elseif (nDim == 3)
       kx = col(rho*sin(theta))*col(repmat(cos(phi),[1 1]))';
       ky = col(rho*sin(theta))*col(repmat(sin(phi),[1 1]))';
       kz = repmat(col(rho*cos(theta)),[1 nRO]);
       
       k_traj = [kx(:) ky(:) kz(:)];
       dens = sqrt(kx(:).^2+ky(:).^2+kz(:).^2);
    end
else
    %uniform sampling
    rho=linspace(-pR,pR,N)';
    
    if (nDim == 2)
        [X,Y]=meshgrid(rho,rho);
        k_traj = [X(:), Y(:)];
    elseif (nDim == 3)
        [X,Y,Z]=meshgrid(rho,rho,rho);
        k_traj = [X(:), Y(:), Z(:)];
    end
    dens = ones(1,size(k_traj,1));
end
%% test kspace
if (nDim ==2)
    dataRadial = sl2d.FourierDomainSignal([k_traj(:,1),k_traj(:,2)]);
elseif (nDim ==3)
    dataRadial = sl3d.FourierDomainSignal([k_traj(:,1),k_traj(:,2),k_traj(:,3)]);
end
dataRadial = dataRadial(:,1)+1i*dataRadial(:,2);

%% scale traj to -0.5 0.5
k_traj = -0.5 + (k_traj+pR)/(2*pR);

dataRadial = dataRadial(1:R:end);
k_traj = k_traj(1:R:end,:);
dens = dens(1:R:end);