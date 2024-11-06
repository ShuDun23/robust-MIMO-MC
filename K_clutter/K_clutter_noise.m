%% Generate K Distribution Clutter + Complex Gaussian Noise
function [Clu, noise, Clu_noise] =K_clutter_noise(Signal,K_Distribute_a,c1,SCNR,CNR)
% shape parameter a ; scale parameter b, b can be calculated by using a

Signal_size = size(Signal);
num = 1;
for i = 1 : length(Signal_size)
    num = num * Signal_size(i);
end
Signal_power = sum(abs(Signal(:)).^2) / num;
s_cn = 10^(SCNR/10);
c_n = 10^(CNR/10);
% c1 = 0.9;
c2 = 1-c1;

%% K-distribution parameters
Simulate_Data_Length=num;
Nfft = 128;

%% Y_k
W1_k = randn(1, Simulate_Data_Length) + 1i * randn(1, Simulate_Data_Length);

F_Index = -0.5 : 1/Nfft : 0.5-1/Nfft; 
Delta_F1 = 0.01;    
Fd1 = 0.1;
Hk1_Abs = exp(-(F_Index-Fd1).^2/(4*(Delta_F1^2)));

Hk_Abs = ifftshift(Hk1_Abs);
[Hk1] = Get_Hk_From_Hk_Abs(Hk_Abs);
hk1 = ifft(Hk1);

Tmp = length(hk1);
Y_k = conv(hk1, W1_k);
Y_k = Y_k(Tmp : end);

K_Distribute_b = sqrt(var(Y_k) / (2*K_Distribute_a));

%% Z_k
W2_k=randn(1,Simulate_Data_Length);

Delta_F2 = 0.0001;
Hk2_Abs = exp(-F_Index.^2./(4*(Delta_F2^2)));
Hk_Abs = ifftshift(Hk2_Abs);
[Hk2] = Get_Hk_From_Hk_Abs(Hk_Abs);
hk2 = ifft(Hk2);

Tmp = length(hk2);
Z_k = conv(hk2, W2_k);
Z_k = Z_k(Tmp : end);
%% Solve the nonlinear equation
S_k = zeros(1, length(Z_k));

S_2_Table = (0 : 0.01 : 200);
Incomplete_Gamma_Result_Table = zeros(1, length(S_2_Table));
E_y2 = var(Y_k);
K_Distribute_Alpha_2 = K_Distribute_b ^2;
for w1 = 1 : 1 : length(S_2_Table)
    tmp = E_y2 * S_2_Table(w1) / (K_Distribute_Alpha_2 * pi);
    Incomplete_Gamma_Result_Table(w1) = gammainc(K_Distribute_a, tmp);
end

Result_Tmp = 1/2 + 1/2 * erf(Z_k/sqrt(2));
S_2_Result = zeros(1, length(Z_k));
for w1 = 1 : 1 : length(Z_k)
    tmp = Result_Tmp(w1);
    tmp1 = abs(Incomplete_Gamma_Result_Table - tmp);
    [~, Index] = min(tmp1);
    S_2_Result(w1) = S_2_Table(Index);
end
S_k = sqrt(S_2_Result);

%% Final result
X_k = Y_k.*S_k;
% figure, subplot(2,1,1),plot(real(X_k));
% title('K-distributed clutter--real part');
% subplot(2,1,2),plot(imag(X_k));
% title('K-distributed clutter--imaginary part');

%% Power
Clutter_Noise_power = Signal_power / s_cn;
Noise_power = Clutter_Noise_power / (c1+c2*c_n);
Clutter_power = c_n * Noise_power;
Clu = sqrt(Clutter_power / mean(abs(X_k(:)).^2)) * X_k;
Clu = reshape(Clu,Signal_size);
%--------------------------------------------------------------------------------
% num_pdf=100;
% maxdat=max(abs(X_k));
% mindat=min(abs(X_k));
% NN=hist(abs(X_k),num_pdf);
% xpdf1=num_pdf*NN/(sum(NN))/(maxdat-mindat);
% xaxis1=mindat:(maxdat-mindat)/num_pdf:maxdat-(maxdat-mindat)/num_pdf;
% th_va1 = 2 ./(K_Distribute_b*gamma(K_Distribute_a)).* ((xaxis1/(2*K_Distribute_b)).^(K_Distribute_a))...
%     .* besselk(K_Distribute_a-1,xaxis1/K_Distribute_b);
% figure;
% plot(xaxis1,xpdf1);
% hold on
% plot(xaxis1,th_va1,':r');
% title('clutter');
% xlabel('magnitude')
% ylabel('pdf')
%--------------------------------------------------------------------------------
noise = sqrt(Noise_power/2) * (randn(Signal_size) + randn(Signal_size)*1i); % complex Gaussian noise

p = c1/1; % the percentage of outliers c1/c2
flag = binornd( 1, p, Signal_size);
Clu_noise = Clu.*(ones(Signal_size) - flag) + noise.*flag;
end