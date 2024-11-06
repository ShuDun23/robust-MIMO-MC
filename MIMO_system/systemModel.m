function [X, S, theta] = systemModel(variable)
%% MIMO radar systemModel
%
%
%% ===============================================================
% Carrier
fc = variable.fc;                           % carrier frequency in Hz
c = variable.c;                             % speed of light (m/s)
Lambda =  c/fc;                             % wavelength (m)
Tpri = 5e-6;                                % pulse duration in sec. pulse repetition interval
% OR = variable.OR;                           % observation ratio
%=========================================================
% Transmit and receive arrays parameters
N = variable.N;                             % Number of samples for each transmitted pulse
Q = variable.Q;                             % number of snapshots
K = variable.K;                             % number of targets
Mt = variable.Mt;                           % number of transmit antennas
Mr = variable.Mr;                           % number of receive antennas
dr = Lambda/2;                              % inter-element spacing at ULA transmitter (m)
dt = Lambda/2;                              % inter-element spacing at ULA receiver (m)
state = variable.state;
% theta = [10, -20, -10, 15, 30, 45, -45, -40, -30, -15, 35]'; % 20 -20 % target 
if matches(state,'MSE')
    % theta = theta(1:K);
    theta = linspace(-60,60,K)';
elseif matches(state,'RES')
    dtheta = variable.dtheta;
    theta = (theta(1):dtheta:(theta(1)+(K-1)*dtheta))';
end
%==========================================================
% Transmit and receive steering matrix
A = exp(1j*2*pi*dt/Lambda* (0:Mt-1)' * sin(theta'/180*pi));  % Transmit steering matrix
B = exp(1j*2*pi*dr/Lambda* (0:Mr-1)' * sin(theta'/180*pi));  % Receive steering matrix
%==========================================================
% Variances of target RCS coefficients
% (to model Swerling 2 RCS fluctuations from pulse to pulse)
% Swerling I not need
Var_mat = linspace(0.1,3,K).';
Var_mat = Var_mat(1:K);
%=========================================================
% Speed of targets in meters per second, typical requirement fD*T<<1
% (fD=2*v/Lambda is the Doppler shift and T is the pulse period)
Speed_mat = linspace(100,600,K); % RES
% Speed_mat = [300 250 320 280 150 200 300 200 130 110]; % MSE
Speed_mat = Speed_mat(1:K);
%=========================================================
% Generation of pulses: Hadamard Pulse
% Orthogonal waveforms S (MtxL) such that (1/L)*S*S' is identity matrix
S=(1/sqrt(2*N))*(hadamard(N)+1j*hadamard(N)); % NxN
S=S(1:Mt, :);
% Propagation
%==========================================================
% % Target reflection coefficients in C
% RCS = ((randn(1,K)+1j*randn(1,K))*diag(sqrt(Var_mat/2)));
% % Vandermonde matrix with Doppler shifts
% Doppler = exp(1j*2*pi/Lambda*2*Speed_mat*Tpri);

% Target reflection coefficients in C (QxK)
% RCS = randn(Q,K)+1j*randn(Q,K); % Complex Normal distribution RCS~CN(0,2) SW I
RCS = (randn(Q,K)+1j*randn(Q,K))*diag(sqrt(Var_mat/2)); % Complex Normal distribution RCS~CN(0,variance) SW II
% Vandermonde matrix with Doppler shifts
Doppler = exp(1j*2*pi/Lambda*(0:Q-1)'*2*Speed_mat*Tpri);

% X = B*diag(RCS(1,:))*diag(Doppler)*A.'*S;  % received signal
X = zeros(Mr, N, Q);
for q = 1:Q
    X(:,:,q) = B*diag(RCS(q,:))*diag(Doppler(q,:))*A.'*S;
end
end