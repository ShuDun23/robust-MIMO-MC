clc % 跑这个需要改所有算法的mse
close all
clear
p = 1;                               % lp的p
[filepath,name,ext] = fileparts(mfilename('fullpath'));
runID = datestr(now,30);
% dBB = -10:2:10;                      % SNR or SCNR
dBB = 10; % 大信噪(杂)比用于对比收敛速度
T = 1; % number of testing 计算MSE和prob of detection, 1个T大概1分钟 T=100 15小时
iter = 1:1:100;
[ERR1,ERR2,ERR3,ERR4] = deal(zeros(T,length(iter)));
K = 1;   % number of target
c_1 = 0.9;
state = 'MSE'; % systemModel里用
noiseType = 'GM'; % GW for Gaussian White; GM for GMM; CLU for clutter
rate = 0.001; % tolerance of detection error  equation(43)
for OR = 0.5    % Observation Rate: 1; 0.2; ...
    for epsilon = 1e-3 % L0-Laplacian Kernel的门限 1e-3 1e-4 1e-7
        tic
        % [Pmusic0,angle0,Pmusic1,angle1,Pmusic2,angle2,Pmusic3,angle3] = deal(zeros(T,18001));
        % [Th, Th0, Th1, Th2, Th3] = deal(zeros(K,length(dBB),T));
        % [bias,bias0,bias1,bias2,bias3] = deal(zeros(K,length(dBB)));
        for t = 1 : T
            variable = struct;
            variable.K = K;                                         % number of targets
            variable.Mt = 10;                                       % number of transmit antennas 调大这两个可以增加music的分辨率
            variable.Mr = 10;                                       % number of receive antennas 这个调大变得很慢 Mr=40 K=30不行
            variable.OR = OR;                                       % observation ratio
            variable.fc = 1e9;                                      % carrier frequency in Hz
            variable.c = 3e8;                                       % speed of light (m/s)
            variable.Lambda = variable.c /variable.fc;              % wavelength (m)
            variable.dr = variable.Lambda/2;                        % inter-element spacing at ULA transmitter (m)
            variable.dt = variable.Lambda/2;                        % inter-element spacing at ULA receiver (m)
            variable.Q = 128;                                       % number of snapshots (pulses)
            variable.N = 128;                                       % number of samples for each transmitted pulse
            variable.state = state;
            %     [X_MF_Omega, OmegaT, X_MF, theta] = systemModel(variable);
            %% System Model
            [X, S, theta] = systemModel(variable); % 纯净的X 三维
            variable.X_data = X;                                    % received signal MrxNxQ
            variable.S = S;                                         % transmitted pulse
            [theta,~] = sort(theta,'descend');
            minDelta = min(theta(1:(K-1))-theta(2:K));
            Noise = X * 0;  % 可以让Noise拥有X一样size的初始化操作
            Clu = X * 0; % Clutter
            CWGN = X * 0; % complex white gaussian noise
            Clu_noise = X * 0;
            for q = 1:variable.Q
                Noise(:,:,q) = (Gaussian_noise(X(:,:,q),'GM',0,c_1,1000) + Gaussian_noise(X(:,:,q),'GM',0,c_1,1000)*1i)/sqrt(2);%!!!!!!!!!!!!!!!!
                % Noise(:,:,q) = noisemix(M,N,a,1,100); %(M,N,a,v1,v2)
            end
            %--------------------------------------------------------------------------------
            for q = 1:variable.Q
                [Clu(:,:,q), CWGN(:,:,q), Clu_noise(:,:,q)] = K_clutter_noise(X(:,:,q),2,c_1,0,10);
            end
            %--------------------------------------------------------------------------------
            [~,fold_ind]=sort(randn(1,numel(X)));        % 采样点index
            Omega_vector = zeros(1, numel(X));           % numel数组元素个数
            Omega_vector(fold_ind(1:(numel(X)*OR))) = 1; % 采样阵 全1 全采样
            OmegaT = reshape(Omega_vector, size(X));     % reshape后的采样阵 三维
            while(prod(reshape(sum(OmegaT,1),1,[])) == 0 || prod(reshape(sum(OmegaT,2),1,[])) == 0) % 确保采样阵(MrxNxQ)里每列、行都至少有一个1
                for n = 1:variable.N % 如果有全零列 或者行 则按列重新生成OmegaT
                    for q = 1:variable.Q
                        [~,fold_ind]=sort(randn(1,variable.Mr));
                        Omega_vector = zeros(variable.Mr, 1);
                        Omega_vector(fold_ind(1:(variable.Mr*OR))) = 1;
                        OmegaT(:,n,q) = Omega_vector;
                    end
                end
            end
            Omega = reshape(OmegaT, variable.Mr*variable.N, variable.Q); % reshape成(MrxN)xQ for matrix completion
            X_RE = reshape(X, variable.Mr*variable.N, variable.Q);       % reshape成(MrxN)xQ for matrix completion
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
                %         [Y_MF, Y_MF_noise, Y_MF_Omega_noise, Omega] = mf(variable);
                X_Omega_noise = X_noise .* OmegaT; % partial sampling 部分采样
                X_RE_Omega_noise = reshape(X_Omega_noise, variable.Mr*variable.N, variable.Q); % reshape成(MrxN)xQ for matrix completion
                %% ===================================
                %% complex matrix completion
                maxiter = 100;
                maxiter1 = 10; % max interation number
                maxiter2 = 100;
                % ====================================
                [Out_X1,~,~,err1,~,iter1] = L0_prox( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter, epsilon, 1e-5);
                % [Out_X2,~,~,~,iter2] = HOW( X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                [Out_X2,~,~,err2,~,iter2] = L0_IQR( X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
                [Out_X3,~,~,~,iter3,~,err3,~] = SVT( X_RE, X_RE_Omega_noise, Omega, maxiter, 1e-5, 0);
                [Out_X4,~,~,err4,~,iter4] = HOW_LMaFit(X_RE, X_RE_Omega_noise, Omega, variable.K, maxiter1, maxiter2, 1e-5);
            end
            [ERR1(t,:),ERR2(t,:),ERR3(t,:),ERR4(t,:)] = deal(err1, err2, err3, err4);
        end
        result_ERR = [mean(ERR1,1);mean(ERR2,1);mean(ERR3,1);mean(ERR4,1)];
        %% ==================================
        close all;
        
        figure
        semilogy(iter,result_ERR(1,:),'o-','Linewidth',1.2)
        hold on
        semilogy(iter,result_ERR(2,:),'x-','Linewidth',1.2)
        semilogy(iter,result_ERR(3,:),'s-','Linewidth',1.2)
        semilogy(iter,result_ERR(4,:),'+-','Linewidth',1.2)
        % legend('basic line', 'Frobineus MC','$\ell_p$-MC','$\ell_0$','Interpreter','latex')
        legend('$\ell_0$-norm','$\ell_0$-IQR','SVT','HOW-IQR','Interpreter','latex')
        xlabel('Iteration number','Interpreter','latex')
        ylabel('RMSE','Interpreter','latex')
        set(gca,'FontSize',12,'FontName','Times');
        grid on

        toc
        % saveas(gcf,['./fig/',name,'_',runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'])
    end
end
%%
% filename = [runID,'_e=',num2str(epsilon),'_OR=',num2str(OR),'_T=',num2str(T),'_r=',num2str(rate),'_',noiseType,'noise','.fig'];
% saveas(gcf,filename);
%%
% save('./data/main_iter GMM.mat')