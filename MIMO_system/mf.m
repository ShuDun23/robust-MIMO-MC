function [Y_MF, Y_MF_noise, Y_MF_Omega_noise, Omega] = mf(variable) % match filter
%% parameters setting
OR = variable.OR;                           % observation ratio
N = variable.N;                             % Number of samples for each transmitted pulse
Q = variable.Q;                             % number of snapshots
Mt = variable.Mt;                           % number of transmit antennas
Mr = variable.Mr;                           % number of receive antennas
X_noise = variable.X_noise;                 % received signal with noise
X = variable.X_data;                        % received signal
S = variable.S;                             % transmitted pulse
%% matched filter
[X_MF, X_MF_noise, X_MF_Omega_noise, OmegaT] = deal(zeros(Mr,Mt,Q));
for q = 1 : Q
    for i = 1 : Mr
        J = binornd( 1, OR, [ 1, Mt ] );
        OmegaT(i,:,q) = J;
        MF = S;
        MF(J==0,:) = 0;
        X_MF_Omega_noise(i,:,q) = 1 * X_noise(i,:,q)*MF';         % Partial Sampling
        X_MF_noise(i,:,q) = 1 * X_noise(i,:,q)*S';                % Full Sampling
        X_MF(i,:,q) = 1 * X(i,:,q)*S';
    end
end
%% reshape matrix
[Y_MF, Y_MF_noise, Y_MF_Omega_noise, Omega] = deal(zeros(Mt*Mr,Q));
for q = 1 : Q
    Y_MF(:,q) = reshape(X_MF(:,:,q),[],1);
    Y_MF_noise(:,q) = reshape(X_MF_noise(:,:,q),[],1);
    Y_MF_Omega_noise(:,q) = reshape(X_MF_Omega_noise(:,:,q),[],1);
    Omega(:,q) = reshape(OmegaT(:,:,q),[],1);
end