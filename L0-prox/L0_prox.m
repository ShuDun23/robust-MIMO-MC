function [M, U, V, RMSE, RMSE_out, iter, time, value] = L0_prox(origin_X, X, Omega_array, rak, maxiter, epsilon, phi)
tic;
[r,c] = size(X);
U = randn(r,rak); % rxTARGET NUMBER
V = randn(rak,c); % no transpose for V
RMSE = zeros(1,maxiter);
[mu, s, S] = Laplacian_kernel(X, epsilon, Inf, 0, phi, Omega_array, 'c'); % 进来先判别
for iter = 1 : maxiter % max iteration number
    for i = 1:r
        col = find(Omega_array(i,:) == 1); % 第i行中为1的列
        V_r = V(:,col);
        x_r = X(i,col);
        n_r = S(i,col);
        U(i,:) = (pinv(2 * V_r * V_r' + phi * eye(rak)) * (2 * V_r * (x_r - n_r)' + phi * U(i,:)'))';
    end
    for j = 1:c
        row = find(Omega_array(:,j) == 1);
        U_c = U(row,:);
        x_c = X(row,j);
        n_c = S(row,j);
        V(:,j) = pinv(2 * U_c' * U_c + phi * eye(rak)) * (2 * U_c' * (x_c - n_c) + phi * V(:,j));
    end
    M = U*V;
    N_Omega = X - M .* Omega_array;
    [mu, s, S] = Laplacian_kernel(N_Omega, epsilon, mu, s, phi, Omega_array, 'c');
    RMSE(iter) = norm((M-origin_X).*Omega_array,'fro') / sqrt(r*c);
    if RMSE(iter) < 0.0001 % 1e-4
        break;
    end
end
RMSE_out = RMSE(iter);
time = toc;
end