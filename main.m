%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for "H. N. Sheng, Z.-Y. Wang, Z. Liu, and H. C. So, 
% "Hybrid ordinary-Welsch function based robust matrix completion for MIMO radar,” 
% IEEE Transactions on Aerospace and Electronic Systems (TAES), 2024."
%
% OUTPUTS:
% RMSE_theta versus SCNR
% Pseudo-spectrum versus theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
close all
clear

[filepath,name,ext] = fileparts(mfilename('fullpath'));
runID = datestr(now,30);
addpath('SVT\')
addpath('K_clutter\')
addpath('MIMO_system\')
addpath('L0-IQR\')
addpath('L0-prox\')

dBB = -10:2:10;
T = 100; % number of testing
[ERR1,ERR2,ERR3,ERR4,ERR5] = deal(zeros(T,length(dBB)));
[MSE, MSE0, MSE1, MSE2, MSE3, MSE4, MSE5] = deal(zeros(T,length(dBB)));
K = 1;   % number of target
c_1 = 0.9;
state = 'MSE';
noiseType = 'GM'; % GM for GMM; CLU for clutter
rate = 0.001; % tolerance of detection error  equation(43) in [27]
for OR = 0.5    % Observation Rate: 1; 0.2; ...
    for epsilon = 1e-3 % L0-Laplacian Kernel threshold
        tic
        [Pmusic0,angle0,Pmusic1,angle1,Pmusic2,angle2,Pmusic3,angle3,Pmusic4,angle4,Pmusic5,angle5] = deal(zeros(T,18001));
        for t = 1 : T
            variable = struct;
            variable.K = K;                                         % number of targets
            variable.Mt = 10;                                       % number of transmit antennas
            variable.Mr = 10;                                       % number of receive antennas
            variable.OR = OR;                                       % observation ratio
            variable.fc = 1e9;                                      % carrier frequency in Hz
            variable.c = 3e8;                                       % speed of light (m/s)
            variable.Lambda = variable.c /variable.fc;              % wavelength (m)
            variable.dr = variable.Lambda/2;                        % inter-element spacing at ULA transmitter (m)
            variable.dt = variable.Lambda/2;                        % inter-element spacing at ULA receiver (m)
            variable.Q = 128;                                       % number of snapshots (pulses)
            variable.N = 128;                                       % number of samples for each transmitted pulse
            variable.state = state;
            %% System Model
            [X, S, theta] = systemModel(variable);
            variable.X_data = X;                                    % received signal MrxNxQ
            variable.S = S;                                         % transmitted pulse
            [theta,~] = sort(theta,'descend');
            minDelta = min(theta(1:(K-1))-theta(2:K));
            Noise = X * 0;
            Clu = X * 0; % Clutter
            CWGN = X * 0; % complex white gaussian noise
            Clu_noise = X * 0;
            for q = 1:variable.Q
                Noise(:,:,q) = (Gaussian_noise(X(:,:,q),'GM',0,c_1,1000) + Gaussian_noise(X(:,:,q),'GM',0,c_1,1000)*1i)/sqrt(2);%!!!!!!!!!!!!!!!!
            end
            %--------------------------------------------------------------------------------
            for q = 1:variable.Q
                [Clu(:,:,q), CWGN(:,:,q), Clu_noise(:,:,q)] = K_clutter_noise(X(:,:,q),2,c_1,0,30);
            end
            %--------------------------------------------------------------------------------
            [~,fold_ind]=sort(randn(1,numel(X)));        
            Omega_vector = zeros(1, numel(X));           
            Omega_vector(fold_ind(1:(numel(X)*OR))) = 1;
            OmegaT = reshape(Omega_vector, size(X));    
            while(prod(reshape(sum(OmegaT,1),1,[])) == 0 || prod(reshape(sum(OmegaT,2),1,[])) == 0)
                for n = 1:variable.N % there cannot be all zero rows or columns
                    for q = 1:variable.Q
                        [~,fold_ind]=sort(randn(1,variable.Mr));
                        Omega_vector = zeros(variable.Mr, 1);
                        Omega_vector(fold_ind(1:(variable.Mr*OR))) = 1;
                        OmegaT(:,n,q) = Omega_vector;
                    end
                end
            end
            Omega = reshape(OmegaT, variable.Mr*variable.N, variable.Q); % reshaped to (MrxN)xQ for matrix completion
            X_RE = reshape(X, variable.Mr*variable.N, variable.Q);       % reshaped to (MrxN)xQ for matrix completion
            [n1,n2] = size(X_RE);
            [mse, mse0, mse1, mse2, mse3, mse4, mse5] = deal(zeros(1,length(dBB)));
            [err1, err2, err3, err4, err5] = deal(zeros(1,length(dBB)));
            %% Partial Sampling
            for no = 1:length(dBB) % noise level
                snr = dBB(no);
                s_n = 10^(snr/10);
                switch noiseType
                    case 'CLU'
                        X_noise = X + Clu_noise / sqrt(s_n); % Clutter + complex WGN
                    case 'GM'
                        X_noise = X + Noise / sqrt(s_n); % complex GMM
                end
                variable.X_noise = X_noise; % received signal with noise
                X_Omega_noise = X_noise .* OmegaT; % partial sampling
                X_RE_Omega_noise = reshape(X_Omega_noise, variable.Mr*variable.N, variable.Q); % reshaped to (MrxN)xQ for matrix completion
                %% ===================================
                %% complex matrix completion
                maxiter = 100;
                maxiter1 = 100; % max interation number
                maxiter2 = 100;
                % l0-prox
                [Out_X1,~,~,~,err1(no),~,~] = L0_prox( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter, epsilon, 1e-5);
                % l0-IQR
                [Out_X2,~,~,~,err2(no),~,~] = L0_IQR( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                % SVT
                [Out_X3,~,~,~,~,~,~,err3(no)] = SVT( X_RE, X_RE_Omega_noise, Omega, maxiter, 1e-5, 0);
                % HOW-IQR
                [Out_X4,~,~,~,err4(no),~,~] = HOW_IQR(X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                % HOW-SASD
                [Out_X5,~,err5(no),~,~,~,~,~,~,~,~] = HOW_SASD(X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter, 2, 2.3);
                %% ==================================
                % DOA estimation
                % complete data without noise
                [Y_MF, Y_MF_noise, ~, ~] = mf(variable);
                variable.X = Y_MF;
                [theta_re,~,~] = MUSIC(variable);

                % complete data + noise -> Basic line
                variable.X = Y_MF_noise;
                [theta_re0,Pmusic0(t,:),angle0(t,:)] = MUSIC(variable);

                % l0-prox [27]
                variable.X_noise = reshape(Out_X1, variable.Mr, variable.N, variable.Q);
                [~, variable.X, ~, ~] = mf(variable);
                [theta_re1,Pmusic1(t,:),angle1(t,:)] = MUSIC(variable);

                % l0-IQR (ours)
                variable.X_noise = reshape(Out_X2, variable.Mr, variable.N, variable.Q);
                [~, variable.X, ~, ~] = mf(variable);
                [theta_re2,Pmusic2(t,:),angle2(t,:)] = MUSIC(variable);

                % SVT [11]
                variable.X_noise = reshape(Out_X3, variable.Mr, variable.N, variable.Q);
                [~, variable.X, ~, ~] = mf(variable);
                [theta_re3,Pmusic3(t,:),angle3(t,:)] = MUSIC(variable);

                % HOW-IQR (ours)
                variable.X_noise = reshape(Out_X4, variable.Mr, variable.N, variable.Q);
                [~, variable.X, ~, ~] = mf(variable);
                [theta_re4,Pmusic4(t,:),angle4(t,:)] = MUSIC(variable);

                % HOW-SASD [30]
                variable.X_noise = reshape(Out_X5, variable.Mr, variable.N, variable.Q);
                [~, variable.X, ~, ~] = mf(variable);
                [theta_re5,Pmusic5(t,:),angle5(t,:)] = MUSIC(variable);
                %% ==================================
                % evaluate performance
                mse(no)  = norm(theta'-theta_re,2)^2/(variable.K);
                mse0(no) = norm(theta'-theta_re0,2)^2/(variable.K);
                mse1(no) = norm(theta'-theta_re1,2)^2/(variable.K);
                mse2(no) = norm(theta'-theta_re2,2)^2/(variable.K);
                mse3(no) = norm(theta'-theta_re3,2)^2/(variable.K);
                mse4(no) = norm(theta'-theta_re4,2)^2/(variable.K);
                mse5(no) = norm(theta'-theta_re5,2)^2/(variable.K);
            end
            [MSE(t,:), MSE0(t,:), MSE1(t,:), MSE2(t,:), MSE3(t,:), MSE4(t,:), MSE5(t,:)] = deal(mse,mse0,mse1,mse2,mse3,mse4,mse5);
        end
        result_MSE = [mean(MSE,1);mean(MSE0,1);mean(MSE1,1);mean(MSE2,1);mean(MSE3,1);mean(MSE4,1);mean(MSE5,1)];
        %% ==================================
        close all;
        figure
        p1=plot(angle0(T,:),Pmusic0(T,:),'.-','color','#77AC30','Linewidth',1.2);p1.MarkerIndices=1:3000:length(angle0);p1.MarkerSize = 10;
        hold on
        p2=plot(angle3(T,:),Pmusic3(T,:),'s-','color','#EDB120','Linewidth',1.2);p2.MarkerIndices=1:3000:length(angle0);
        p3=plot(angle1(T,:),Pmusic1(T,:),'o-','color','#0072BD','Linewidth',1.2);p3.MarkerIndices=1:3000:length(angle0);
        p4=plot(angle2(T,:),Pmusic2(T,:),'x-','color','#D95319','Linewidth',1.2);p4.MarkerIndices=1:3000:length(angle0);
        p5=plot(angle5(T,:),Pmusic5(T,:),'^-','color',"#FF00FF",'Linewidth',1.2);p5.MarkerIndices=1:3000:length(angle0);
        p6=plot(angle4(T,:),Pmusic4(T,:),'+-','color','#7E2F8E','Linewidth',1.2);p6.MarkerIndices=1:3000:length(angle0);
        % legend('basic line', 'Frobineus MC','$\ell_p$-MC','$\ell_0$','Interpreter','latex')
        legend('Basic Line','SVT','$\ell_0$-prox','$\ell_0$-IQR','HOW-SASD','HOW-IQR','Interpreter','latex','Location','southwest')
        xlabel('$\theta$','Interpreter','latex')
        ylabel('Pseudo-spectrum (dB)','Interpreter','latex')
        set(gca, 'XTick', -90:30:90)
        set(gca,'FontSize',12,'FontName','Times');
        grid on

        figure
        semilogy(dBB,sqrt(result_MSE(2,:)),'.-','color','#77AC30','Linewidth',1.2);
        hold on
        semilogy(dBB,sqrt(result_MSE(5,:)),'s-','color','#EDB120','Linewidth',1.2)
        semilogy(dBB,sqrt(result_MSE(3,:)),'o-','color','#0072BD','Linewidth',1.2)
        semilogy(dBB,sqrt(result_MSE(4,:)),'x-','color','#D95319','Linewidth',1.2)
        semilogy(dBB,sqrt(result_MSE(7,:)),'^-','color',"#FF00FF",'Linewidth',1.2)
        semilogy(dBB,sqrt(result_MSE(6,:)),'+-','color','#7E2F8E','Linewidth',1.2)
        % legend('basic line', 'Frobineus MC','$\ell_p$-MC','$\ell_0$','Interpreter','latex')
        legend('Basic Line','SVT','$\ell_0$-prox','$\ell_0$-IQR','HOW-SASD','HOW-IQR','Interpreter','latex','Location','southwest')
        xlabel('SCNR (dB)','Interpreter','latex')
        ylabel('$\mathrm{RMSE}_\theta$','Interpreter','latex')
        set(gca,'FontSize',12,'FontName','Times');
        grid on

        toc
        % saveas(gcf,['./fig/',name,'_',runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'])
    end
end
%%
% filename = [runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'];
% saveas(gcf,filename);