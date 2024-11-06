function [Hk] = Get_Hk_From_Hk_Abs(Hk_Abs)

N = length(Hk_Abs);
Hk = zeros(1, N);

% N 为偶数时
if(mod(N, 2) == 0)
    for w1 = 0 : 1 : N/2 - 1
        Hk(1, w1+1) = Hk_Abs(1, w1+1) * exp((-j*2*pi/N) * w1 * (N-1)/2);
    end;
    Hk(N/2) = 0;
    for w1 = N/2+1 : 1 : N-1
        Hk(1, w1+1) = Hk_Abs(1, w1+1) * exp((j*2*pi/N) * (N-w1) * (N-1)/2);
    end;
else
     %  如果是奇数
     for w1 = 0 : 1 : (N-1)/2
        Hk(1, w1+1) = Hk_Abs(1, w1+1) * exp((-j*2*pi/N) * w1 * (N-1)/2);
    end;
    for w1 = (N+1)/2 : 1 : N-1
        Hk(1, w1+1) = Hk_Abs(1, w1+1) * exp((j*2*pi/N) * (N-w1) * (N-1)/2);
    end;
end;    

    
    