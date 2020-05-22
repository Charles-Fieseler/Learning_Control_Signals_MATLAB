function [ coeff, Omega, Phi, romA, U, S, V ] = dmd( dat, dt,...
    r, r_percent, useScaling, useFBDMD, useTLS) %Optional parameters
% Dynamic Mode Decomposition
%   This program breaks a data matrix into modes and their temporal
% frequencies, assuming a Fourier-type basis for the temporal dynamics  
%
% INPUTS
%   dat - Data matrix (columns are time slices, rows are spatial)... should
%       be all real
%   dt = deltaT
%   r = (optional) the order to approximate the data to
%   r_percent - (optional) a percentage to automatically decrease the order
%               of the data to capture (note: helps with singular matrices)
%   useScaling - if the returned modes should have unit norm (default) or
%                be scaled by the eigenvalues
%   useFBDMD - if we use forward-backward DMD, a simple extension for
%              stabilizing the algorithm
%
% OUTPUTS - 
%   coeff - coefficients of the basis modes
%   omega - frequency of the basis modes
%   modes - DMD basis modes (spatial dynamics)
%
%
% Dependencies
%   Other m-files required: 
%   Subfunctions: 
%   MAT-files required: 
%
%   See also: OTHER_FUNCTION_NAME
%
%
%
% Author: Charles Fieseler
% University of Washington, Dept. of Physics
% Email address: charles.fieseler@gmail.com
% Website: coming soon
% Created: 27-Feb-2017
%========================================


%% Defaults
if ~exist('dt', 'var') || isempty(dt)
    dt = 1;
end
if ~exist('useScaling','var') || isempty(useScaling)
    useScaling = false;
end
if ~exist('useFBDMD','var') || isempty(useFBDMD)
    useFBDMD = false;
end
if ~exist('useTLS','var') || isempty(useTLS)
    useTLS = false;
end
%==========================================================================


%% ROM-DMD

%---------------------------------------------
% Break the data into submatrices
%---------------------------------------------
% For a matrix with M time slices, we want two submatrices of slices 1 to
% M-1, and 2 to M
subDat1 = dat(:,1:end-1);
subDat2 = dat(:,2:end);

%---------------------------------------------
% SVD
%---------------------------------------------
% For large data matrices, we want to do a low-rank DMD, as we won't be
% able to deal with the full matrix
[U,S,V] = svd(subDat1,'econ');
if useFBDMD
    %Calculate the backwards propagator as well
    %   switch the role of subDat1 and subDat2
%     error('This mode is extremely unstable')
    [U2,S2,V2] = svd(subDat2,'econ');
end

%Throw away all but the top r modes
if ~exist('r','var') || r==-1
    r = size(U,2);
elseif r == 0
    r = optimal_truncation(subDat1);
end
if exist('r_percent','var')
    if r_percent<1
        sigs = diag(S);
        sigs = cumsum(sigs)/sum(sigs);
        r = max( find(sigs>r_percent,1)-1, 1);
    else
        r = size(U,2);
    end
end
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
if useFBDMD
    U2 = U2(:,1:r);
    S2 = S2(1:r,1:r);
    V2 = V2(:,1:r);
end

%---------------------------------------------
% ROM propagator
%---------------------------------------------
%Get the reduced order model propagator, which uses SVD modes as the first
%reduction
romA = (U')*subDat2*V/S;
if useFBDMD
    romA2 = (U2')*subDat1*V2/S2;
    
    %Improved estimate for A
    %   Note: there are ambiguities for the matrix sqrt...
    romA = sqrtm(romA/romA2);
%     romA = (romA / romA2) ^ 0.5;
%     [V, D] = eig(romA, 'vector');
%     [~, D2] = eig(romA2, 'vector');
%     romA = V*diag(sqrt(D./D2))/V;
end

%---------------------------------------------
% Get frequencies and modes
%---------------------------------------------
%The DMD basis modes and frequencies are nearly the eigenvectors and eigenvalues
%of the reduced order propagator
if ~useScaling
    [W, V] = eig(romA);
else
    S_half = S^(1/2);
    romA_hat = (S_half^-1)*romA*S_half;
    [W_hat, V_hat] = eig(romA_hat);
    V = V_hat;
    W = S_half*W_hat;
end

%The basis modes are the eigenvectors projected back onto the data
Phi = U*W;
%Phi = subDat2*V*(S^-1)*W;

%The frequencies are directly related to the eigenvalues
Omega = log(diag(V))/dt;

%Coefficients are simply the initial amplitudes
coeff = Phi\dat(:,1);

%==========================================================================
end

