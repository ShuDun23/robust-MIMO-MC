%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for "H. N. Sheng, Z.-Y. Wang, Z. Liu, and H. C. So, 
% "Hybrid ordinary-Welsch function based robust matrix completion for MIMO radar,” 
% IEEE Transactions on Aerospace and Electronic Systems (TAES), 2024."

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M_out, U, V, RMSE_2, RMSE_2_out, iter, time] = HOW_IQR(origin_X, X, Omega_array, rank, maxiter1, maxiter2, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
% -X: origin_X with missing entries and noise
% -Omega_array：randomly sampling matrix consisting of 0 and 1
% -rank：rank of origin_X
% -maxiter1：max iteration number1 100
% -maxiter2：max iteration number2 100
% -lambda: 1e-5 for proximal BCD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
% -M: M=UV
% -U
% -V
% -RMSE
% -iter
% -time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[m,n] = size(X); % row col
U = randn(m,rank); % rxTARGET NUMBER
V = randn(rank,n); % no transpose for V
Z = U * V - U * V .* Omega_array;
S = Z * 0;
RMSE_1 = zeros(1,maxiter1);
RMSE_2 = zeros(1,maxiter2);

xi1 = 2;
xi2 = xi1*sqrt(2);
c = 2;
sigma = 1;

for iter = 1 : maxiter1 % max iteration number
    G = X + Z - S;
    U = (G * V' - lambda * U) * inv(V * V' - lambda * eye(rank));
    V = inv(U' * U - lambda * eye(rank)) * (U' * G - lambda * V);
    M = U * V;
    Z = M - M.*Omega_array;    
end

D = X + Z - M;  % X-UV, X - M.*Omega_array
D_m_n = D(find(D));
d = iqr(abs(D_m_n))/1.349;

c = min([c xi1*d]);

sigma = min([sigma xi2*d]);

ONE_1=ones(m,n);
ONE_1(abs(D)<c)=0;
pphiD = D - D .* exp((c^2-abs(D).^2) / (sigma^2));
%--------------------------------------

S = pphiD .* ONE_1.*Omega_array;

for iter = 1 : maxiter2
    G = X + Z - S;
    U = (G * V' - lambda * U) * inv(V * V' - lambda * eye(rank));
    V = inv(U' * U - lambda * eye(rank)) * (U' * G - lambda * V);
    Z = U * V - U * V.*Omega_array;

    M = U * V;

    D = X + Z - M;
    D_m_n = D(find(D));
    d = iqr(abs(D_m_n))/1.349;
    c = min([c xi1*d]);
    %-----------------------------------------------------------
    sigma = min([sigma xi2*d]);

    ONE_1=ones(m,n);
    %-------------------------------------------------
    ONE_1(abs(D)<c)=0;
    %-------------------------------------------------
    pphiD = D - D .* exp((c^2-abs(D).^2) / sigma^2);
    S = pphiD .* ONE_1.*Omega_array;

    % stop condition
    RMSE_2(iter)= norm((origin_X - (M - Z)).*Omega_array,'fro') / sqrt(m*n);

    if RMSE_2(iter) < 0.0001 % 1e-4
        break;
    end
end
M_out = M;
RMSE_2_out = RMSE_2(iter);
time = toc;
end