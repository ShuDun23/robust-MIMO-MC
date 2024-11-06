function [theta_re,Pmusic,angle] = MUSIC(variable)

if variable.state == '1'
X = variable.X;
Y = zeros(variable.Mt*variable.Mr,variable.Q);
for q = 1 : variable.Q
    Y(:,q) = reshape(X(:,:,q),[],1);
end
else 
    Y = variable.X;
end
Rxx=Y*Y'/variable.Q;                % covariance matrix
[EV,D]=eig(Rxx);
% EVA=diag(D)';                     % ????????????????
[~,I]=sort(diag(D));                % sort the eigenvalues
EV=fliplr(EV(:,I));                 % sort the eigenvectors
res = 0.01;
len = 180 / res + 1;
[angle, Pmusic] = deal(zeros(1,len));

% traverse angles in 0.1 unit and compute spectrum
for iang = 1:len
    angle(iang) = (iang - 1) * res - 90;
    A = exp(1j*2*pi*variable.dt/variable.Lambda* (0:variable.Mt-1)' * sin(angle(iang)'/180*pi));  % Transmit steering matrix
    B = exp(1j*2*pi*variable.dr/variable.Lambda* (0:variable.Mr-1)' * sin(angle(iang)'/180*pi));  % Receive steering matrix
    V = kron(A,B);
    En=EV(:,variable.K+1:variable.Mt*variable.Mr);                   % noise subspace constructed by the columns from the K+1th to Nth
    Pmusic(iang)=1/(V'*En*En'*V);
end
Pmusic=abs(Pmusic);
Pmmax=max(Pmusic);
Pmusic=10*log10(Pmusic/Pmmax);            % normalization procedure

h=plot(angle,Pmusic);
set(h,'Linewidth',2);
xlabel('$\theta$','Interpreter','latex');
ylabel('Max Power Spectrum/(dB)');
set(gca, 'XTick', -90:30:90);
grid on;

[peaks,locs]=findpeaks(Pmusic);
[~,ind] = sort(peaks,'descend');
maxPeaksLocs = locs(ind(1:variable.K));
theta_re = angle(maxPeaksLocs);
[theta_re,~] = sort(theta_re,'descend');
end