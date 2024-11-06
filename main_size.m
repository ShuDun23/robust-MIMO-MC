%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for "H. N. Sheng, Z.-Y. Wang, Z. Liu, and H. C. So, 
% "Hybrid ordinary-Welsch function based robust matrix completion for MIMO radar,” 
% IEEE Transactions on Aerospace and Electronic Systems (TAES), 2024."
% 
% OUTPUTS:
% RMSE and runtime versus matrix size
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

dBB = 10;
T = 100; % number of testing
Q = 64:64:512;
K = 1;
[ERR1,ERR2,ERR3,ERR4,ERR5,TIME1,TIME2,TIME3,TIME4,TIME5] = deal(zeros(T,length(Q)));
c_1 = 0.9;
state = 'MSE';
noiseType = 'GM'; % GM for GMM; CLU for clutter
rate = 0.001; % tolerance of detection error  equation(43) in [27]
for OR = 0.5    % Observation Rate: 1; 0.2; ...
    for epsilon = 1e-3 % L0-Laplacian Kernel threshold
        % tic
        for t = 1 : T
            [err1,err2,err3,err4,err5,time1,time2,time3,time4,time5] = deal(zeros(1,length(Q)));
            for q1 = 1:length(Q)
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
                variable.Q = Q(q1);                                     % number of snapshots (pulses)
                variable.N = 128;                                       % number of samples for each transmitted pulse
                variable.state = state;
                %% System Model
                [X, S, theta] = systemModel(variable);
                variable.X_data = X;                                    % received signal MrxNxQ
                variable.S = S;                                         % transmitted pulse
                [theta,~] = sort(theta,'descend');
                minDelta = min(theta(1:(variable.K-1))-theta(2:variable.K));
                Noise = X * 0;
                Clu = X * 0; % Clutter
                CWGN = X * 0; % complex white gaussian noise
                Clu_noise = X * 0;
                for q = 1:variable.Q
                    Noise(:,:,q) = (Gaussian_noise(X(:,:,q),'GM',0,c_1,1000) + Gaussian_noise(X(:,:,q),'GM',0,c_1,1000)*1i)/sqrt(2);
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
                    for n = 1:variable.N 
                        for q = 1:variable.Q
                            [~,fold_ind]=sort(randn(1,variable.Mr));
                            Omega_vector = zeros(variable.Mr, 1);
                            Omega_vector(fold_ind(1:(variable.Mr*OR))) = 1;
                            OmegaT(:,n,q) = Omega_vector;
                        end
                    end
                end
                Omega = reshape(OmegaT, variable.Mr*variable.N, variable.Q);
                X_RE = reshape(X, variable.Mr*variable.N, variable.Q);
                [n1,n2] = size(X_RE);
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
                    X_RE_Omega_noise = reshape(X_Omega_noise, variable.Mr*variable.N, variable.Q);
                    %% ===================================
                    %% complex matrix completion
                    maxiter = 100;
                    maxiter1 =  100     ; % max interation number
                    maxiter2 =  100;
                    [Out_X1,~,~,~,err1(q1),iter1,time1(q1)] = L0_prox( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter, epsilon, 1e-5);
                    [Out_X2,~,~,~,err2(q1),iter2,time2(q1)] = L0_IQR( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                    [Out_X3,~,~,~,iter3,time3(q1),~,err3(q1)] = SVT( X_RE, X_RE_Omega_noise, Omega, maxiter, 1e-4, 0);
                    [Out_X4,~,~,~,err4(q1),iter4,time4(q1)] = HOW_IQR(X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                    [Out_X5,~,err5(q1),~,~,~,~,~,~,~,time5(q1)] = HOW_SASD(X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter, 2, 2.3);
                end
            end
            [ERR1(t,:),ERR2(t,:),ERR3(t,:),ERR4(t,:),ERR5(t,:)] = deal(err1, err2, err3, err4, err5);
            [TIME1(t,:),TIME2(t,:),TIME3(t,:),TIME4(t,:),TIME5(t,:)] = deal(time1, time2, time3, time4, time5);
        end
        result_ERR = [mean(ERR1,1);mean(ERR2,1);mean(ERR3,1);mean(ERR4,1);mean(ERR5,1)];
        result_TIME = [mean(TIME1,1);mean(TIME2,1);mean(TIME3,1);mean(TIME4,1);mean(TIME5,1)];
        %% ==================================
        close all;

        figure
        semilogy(Q,result_ERR(3,:),'s-','color',"#EDB120",'Linewidth',1.2)
        hold on
        semilogy(Q,result_ERR(1,:),'o-','color',"#0072BD",'Linewidth',1.2)
        semilogy(Q,result_ERR(2,:),'x-','color',"#D95319",'Linewidth',1.2)
        semilogy(Q,result_ERR(5,:),'^-','color',"#77AC30",'Linewidth',1.2)
        semilogy(Q,result_ERR(4,:),'+-','color',"#7E2F8E",'Linewidth',1.2)
        % legend('basic line', 'Frobineus MC','$\ell_p$-MC','$\ell_0$','Interpreter','latex')
        legend('SVT','$\ell_0$-prox','$\ell_0$-IQR','HOW-SASD','HOW-IQR','Interpreter','latex','Location','southeast')
        xlabel('Size','Interpreter','latex')
        ylabel('RMSE','Interpreter','latex')
        set(gca,'FontSize',12,'FontName','Times');
        set(gca,'XTick',[64:64:512]);
        xlim([64,512]);
        grid on
        
        figure
        plot(Q,result_TIME(3,:),'s-','color',"#EDB120",'Linewidth',1.2)
        hold on
        plot(Q,result_TIME(1,:),'o-','color',"#0072BD",'Linewidth',1.2)
        plot(Q,result_TIME(2,:),'x-','color',"#D95319",'Linewidth',1.2)
        plot(Q,result_TIME(5,:),'^-','color',"#77AC30",'Linewidth',1.2)
        plot(Q,result_TIME(4,:),'+-','color',"#7E2F8E",'Linewidth',1.2)
        % legend('basic line', 'Frobineus MC','$\ell_p$-MC','$\ell_0$','Interpreter','latex')
        legend('SVT','$\ell_0$-prox','$\ell_0$-IQR','HOW-SASD','HOW-IQR','Interpreter','latex','Location','southeast')
        xlabel('Size','Interpreter','latex')
        ylabel('Running time (s)','Interpreter','latex')
        set(gca,'FontSize',12,'FontName','Times');
        set(gca,'XTick',[64:64:512]);
        xlim([64,512]);
        grid on

        % toc
        % saveas(gcf,['./fig/',name,'_',runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'])
    end
end
%%
% filename = [runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'];
% saveas(gcf,filename);