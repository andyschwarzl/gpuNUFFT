function [k_traj,dataRadial,dens] = create_phantom(N, radialTraj, nDim, n3D)  

if nargin < 4
    n3D = 1;
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
rho=linspace(-pR,pR,N)';

if (radialTraj)
    numSpokes = N/2;
    nRO = N/2;

    % Trajectory
    theta=linspace(0,pi-pi/numSpokes,numSpokes);
    phi=linspace(-pi,pi-pi/nRO,nRO)';

    if (nDim == 2)
        kx = rho*cos(theta);
        ky = rho*sin(theta);
    
        k_traj = [kx(:) ky(:)]';
    elseif (nDim == 3)
       kx = col(rho*sin(theta))*col(repmat(cos(phi),[1 nRO]))';
       ky = col(rho*sin(theta))*col(repmat(sin(phi),[1 nRO]))';
       kz = repmat(col(rho*cos(theta)),[1 nRO*nRO]);
       
       k_traj = [kx(:) ky(:) kz(:)]';
    end
    dens = ones(1,size(k_traj,2));
else
    %uniform sampling
    [X,Y,Z]=meshgrid(rho,rho,rho);
    if (nDim == 2)
        k_traj = [X(:), Y(:)];
    elseif (nDim == 3)
        k_traj = [X(:), Y(:), Z(:)];
    end
    dens = ones(1,size(k_traj,2));
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
