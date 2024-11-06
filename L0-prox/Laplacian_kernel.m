function [mu, s, S] = Laplacian_kernel(X, epsilon, mu, s, phi, Omega_array, mode)
%%
% To find the outliers in noise.
% s是o: O的列
% S是O: Outlier阵
%%
n = X(Omega_array == 1);
S = Omega_array * 0;
switch mode
    case 'r' % real
        noise = [real(n);imag(n)];
        A_E = std(noise);%标准差%A_E表示σE
        IQR = iqr(noise);%(prctile(Noise,75) - prctile(Noise,25))
        sigma = 1.06*min(A_E,IQR/1.34)*length(noise)^-0.2;
        w = exp(-abs((noise)/sigma))/(2*sigma);%P表示权重 before 2*pi*sqrt(A)
        mu = min([abs(noise(w<epsilon)).^2; mu]);
        s = (noise + phi * s / 2) / (1 + phi / 2) .* (abs(noise).^2 + phi * abs(s).^2  / 2 - phi * abs(noise - s).^2 / (phi + 2) >= mu);
        S(Omega_array == 1) = s(1:length(s)/2) + 1i * s(length(s)/2+1:length(s));
    case 'c' % complex
        noise = n;
        sigma_r = 1*1.4826*median(abs(real(noise)-median(real(noise))));
        sigma_i = 1*1.4826*median(abs(imag(noise)-median(imag(noise))));
        w = exp(-abs((real(noise))/sigma_r))/(2*sigma_r).*exp(-abs((imag(noise))/sigma_i))/(2*sigma_i); % real
        mu = min([abs(noise(w<epsilon)).^2; mu]); % real 反应了outlier的模
        s = (noise + phi * s / 2) / (1 + phi / 2) .* (abs(noise).^2 + phi * abs(s).^2  / 2 - phi * abs(noise - s).^2 / (phi + 2) >= mu);
        S(Omega_array == 1) = s;
end
end