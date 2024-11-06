function [M_out, U, V, RMSE_2, RMSE_2_out, iter, time] = L0_IQR(origin_X, X, Omega_array, rank, maxiter1, maxiter2, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
% -X: origin_X with missing entries and noise
% -Omega_array：randomly sampling matrix consisting of 0 and 1
% -rank：rank of origin_X
% -maxiter1: max iteration number1 100
% -maxiter2: max iteration number2 100
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
Z = randn(m,n);
S = Omega_array * 0;
RMSE_1 = zeros(1,maxiter1);
RMSE_2 = zeros(1,maxiter2);

xi1 = 2;
c = 2;

for iter = 1 : maxiter1 % max iteration number
    G = X + Z - S;
    U = (G * V' - lambda * U) * pinv(V * V' - lambda * eye(rank));
    V = pinv(U' * U - lambda * eye(rank)) * (U' * G - lambda * V);
    Z = U * V - U * V.*Omega_array;

    M = U * V;
end

D = X + Z - M;
D_m_n = D(find(D));
d = iqr(abs(D_m_n))/1.349;

c = min([c xi1*d]); 

S = 0.*(abs(D)<=c)+ D.*(abs(D)>c);

for iter = 1 : maxiter2
    
    G = X + Z - S;
    U = (G * V' - lambda * U) * pinv(V * V' - lambda * eye(rank));
    V = pinv(U' * U - lambda * eye(rank)) * (U' * G - lambda * V);
    Z = U * V - U * V.*Omega_array;

    M = U * V;

    D = X + Z - M;  % X-UV X - M.*Omega_array
    D_m_n = D(find(D));

    d = iqr(abs(D_m_n))/1.349;
    c = min([c xi1*d]);

    S = 0.*(abs(D)<=c)+ D.*(abs(D)>c);

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