function [Out_X, RMSE, RMSE_OUT, T_iter,loss_HOW, rel_1,S,RMSE_Omega,U,V,time] = HOW_SASD(M,M_Omega,Omega_array,rak,maxiter,ip,ieta)
% This matlab code implements the Max-Dis
% method for Matrix Completion.
%
% M_Omega - m x n observed matrix
% Omega_array is the subset
% rak is the rank of the object matrix
% maxiter - maximum number of iterations
tic
f_welsch=@(x,c,sigma) 0.5*x.^2.*(abs(x)<=c)+(sigma^2/2*(1-exp((c^2-x.^2)/sigma^2))+c^2/2).*(abs(x)>c);
loss_HOW = 0.1;
[m,n] = size(M_Omega);
RMSE = [];
RMSE_Omega = [];
sum_Omega = sum(Omega_array(:));
S = 0;
sigma =1000;
rel_1 = [];
L_f_1 = 0;
T_iter = [];
T_iter = [T_iter 0];
th = 10^(-4);
scale = 1000;
% Initializing U and V
U =randn(m,rak);
V =randn(rak,n);
X_1 = 0;

Z = zeros(m,n);
lambda = 10^(-5);
for k = 1:100

    G = M_Omega + Z - S;
    U = (G * V' - lambda * U) * inv(V * V' - lambda * eye(rak));
    V = inv(U' * U - lambda * eye(rak)) * (U' * G - lambda * V);
    X = U * V;
    Z = X - X.*Omega_array;

end

for iter = 1 : maxiter

    L_f = M_Omega - X.*Omega_array - S;
    rel = norm(L_f-L_f_1,'fro')^2/norm(L_f_1,'fro')^2;
    L_f_1 = L_f;


    T = M_Omega - X.*Omega_array;
    t_m_n = abs(T(find(T)));
    loc_1 = iqr(t_m_n)/1.349;


    sigma = min([sigma ip*loc_1]);

    scale = min([scale ieta*loc_1]);

    T(abs(T)-sigma<0)=0;
    S = T.*(1-exp((sigma^2-abs(T).^2)/(scale^2)));
    RMSE= [RMSE norm((M-X).*Omega_array,'fro')/sqrt(m*n)];
    RMSE_OUT=RMSE(end);

    D = (M_Omega - S);

    dU = -(D-Omega_array.*(U*V))*V';
    du=-dU*(inv(V*V'));
    tu=-trace(dU'*du)/(norm(Omega_array.*(du*V),'fro'))^2;
    U=U+tu*du;

    dV=-U'*((D-Omega_array.*(U*V)));
    dv=inv(U'*U)*dV;
    tv=-trace(dV'*dv)/(norm(Omega_array.*(U*dv),'fro'))^2;
    V=V+tv*dv;
    X=U*V;

    loss_HOW = [loss_HOW sum(sum(f_welsch(M_Omega - X.*Omega_array,sigma,scale)))];
    if abs(loss_HOW(end)-loss_HOW(end-1))/loss_HOW(end-1)<0.0001
        break
    end
end
Out_X = X;
time = toc;
end
