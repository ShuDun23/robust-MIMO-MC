function [OUT, U, Sigma, V, numiter, out_time, MSE, MSE_out]  = SVT(M,X,Omega,maxiter,tol,EPS)
% Inputs:
%   M - Original M (complete)
%   n - size of the matrix X assumed n(1) by n(2). If n is a single integer, it
% is understood that n(1) = n(2).
n = size(M);
%
%   Omega - set of observed entries.  Should be linearly indexed. A 2xm matrix Stores
%   all the linear indexes.
Omega_re = find(Omega);
%
%   b - data vector of the form M(Omega)
b = X(Omega_re);
%
%   tau - parameter defining the objective functional 
tau = 0.0044*sqrt(n(1)*n(2));
%
%   delta - step size.  Choose delta less than 2 to be safe but
%       conservative; choose delta closer to n(1)*n(2)/length(Omega)
%       to be riskier (i.e. algorithm may diverge)
delta = 1.2 * n(1)*n(2)/length(Omega_re);
%
%   maxiter - maximum number of iterations
%
%   tol - stopping criteria (default: 1e-4)
%
%   EPS - noise constraint.  This relaxes the constraints, so that they
%       are now of the form | X(i,j) - M(i,j) | <= EPS,
%       for all indices (i,j) in omega.  Default: 0
%
% Outputs: matrix X stored in SVD format X = U*diag(S)*V'
% 
%   U - n1xr left singular vectors 
% 
%   S - rx1 singular values
%
%   V - n2xr right singular vectors 
%
%   numiter - number of iterations to achieve convergence
%
%   output - a structure with data from each iteration.  Includes:
%       output.nuclearNorm  - nuclear norm of current iterate
%       output.rank         - rank of current iterate
%       output.time         - time taken for one iteraration
%       output.residual     - the relative residual, norm(x-b)/norm(b)


global VERBOSE
if isempty(VERBOSE)
    % -- feel free to change these 'verbosity' parameters
    % VERBOSE = false;
    VERBOSE = 1;    % a little bit of output
    % VERBOSE = 2;    % even more output
end

time1 = cputime;
tic;

if nargin < 6 || isempty(EPS)
    EPS = false;
end
if nargin < 5 || isempty(tol)
    tol = 1e-4;
end
if nargin < 4 || isempty(maxiter)
    maxiter = 500;
end
    
if length(n) == 1,
    n1 = n(1); n2 = n1;
elseif length(n) == 2,
    n1 = n(1); n2 = n(2);
end
if n1*n2 < 100*100, SMALLSCALE = true; else SMALLSCALE = true; end

m = length(Omega_re); [temp,indx] = sort(Omega_re);
% simpler: sort b also
incre = 5;
normb = norm(b);

[i, j] = ind2sub([n1,n2], Omega_re);
USE_SLOW_UPDATE     = false;
if EPS
    % with inequality constraints, should take delta = delta/sqrt(2) at
    % least, or delta = delta/2
    delta = delta/sqrt(2);
%     delta = delta/2; tau = 2*tau;
%     y1 = max(b-EPS,0); y2 = max(-b-EPS,0); % doesn't work well
    y1 = max(b,0); y2 = max(-b,0);
    Y = sparse(i,j,y1-y2,n1,n2,m);
    normProjM = normest(Y,1e-2);
    k0 = ceil(tau/(delta*normProjM));
    y1 = k0*delta*y1;
    y2 = k0*delta*y2;
    try
        updateSparse(Y,y1-y2,indx);
    catch
        l = lasterror;
        if strcmpi( l.identifier, 'MATLAB:UndefinedFunction')
            % mex file not installed, so do this instead:
            [indx_i,indx_j,s] = find(Y);
            Y = updateSparse_slow(Y,y1-y2,indx,indx_i,indx_j);
            USE_SLOW_UPDATE     = true;
        else
            % some other error (unexpected)
            rethrow(lasterror)
        end
    end
else
    Y = sparse(i,j,b,n1,n2,m);
    normProjM = normest(Y,1e-2);
    k0 = ceil(tau/(delta*normProjM));
    normb = norm(b);
    y = k0*delta*b; % kicking by k0 steps
    try
        updateSparse(Y,y,indx);
    catch
        l = lasterror;
        if strcmpi( l.identifier, 'MATLAB:UndefinedFunction')
            % mex file not installed, so do this instead:
            [indx_i,indx_j,s] = find(Y);
            Y = updateSparse_slow(Y,y,indx,indx_i,indx_j);
            USE_SLOW_UPDATE     = true;
        else
            % some other error (unexpected)
            rethrow(lasterror)
        end
    end
    
end
r = 0;

out.residual = zeros(maxiter,1);
out.rank= zeros(maxiter,1);
out.time = zeros(maxiter,1);
out.nuclearNorm = zeros(maxiter,1);

% What the best way to multiply a sparse matrix?
[forwardType, transposeType] = findBestMultiply(Y,.2);


if VERBOSE==1, fprintf('\nIteration:   '); end
for k = 1:maxiter
    if VERBOSE==1, fprintf('\b\b\b\b%4d',k);  end
    s = r + 1;
    
    rInc = 4;  % make this larger for more accuracy
    %if tol < 1e-4  && relRes < 1e-1
    %rInc = rInc + max( round(log10( abs(1e-1/relRes) )), 5 );
    %end
    s = min( [r + rInc, n1, n2] );

    if SMALLSCALE
        [U,Sigma,V] = svd(full(Y));
    else
        % Make routines for multiplying by a sparse matrix
        Yt = Y';
        switch forwardType
            case 1, Yforward = @(x) Y*x;
            case 2, Yforward = @(x) Yt'*x;
            case 3, Yforward = @(x) smvp(Y,x);
        end
        switch transposeType
            case 1, Ytranspose = @(x) Yt*x;
            case 2, Ytranspose = @(x) Y'*x;
            case 3, Ytranspose = @(x) smvp(Yt,x);
        end
        OK = 0;
        while ~OK
            opts = [];
            if ~isreal(b), opts.eta = 1e-16; end
            [U,Sigma,V] = lansvd(Yforward,Ytranspose,n1,n2,s,'L',opts);
            %[U,Sigma,V] = lansvd(Y,s,'L');
            OK = (Sigma(s,s) <= tau) || ( s == min(n1,n2) );
            s = min(s + incre, min(n1,n2));
        end
    end
   
    sigma = diag(Sigma); r = sum(sigma > tau);
    U = U(:,1:r); V = V(:,1:r); sigma = sigma(1:r) - tau; Sigma = diag(sigma);
    
    x = XonOmega(U*diag(sigma),V,Omega_re);
    eTime = cputime - time1;
    if VERBOSE == 2
        fprintf('iteration %4d, rank is %2d, rel. residual is %.1e\n',k,r,norm(x-b)/normb);
    end
    relRes = norm(x-b)/normb;
    out.residual(k) = relRes;
    out.time(k) = eTime;
    out.rank(k) = r;
    out.nuclearNorm(k) = sum(sigma);

    time1 = cputime;
    
%     if (relRes < tol)
%         break
%     end
    if EPS && norm(x-b,'inf') < 2*EPS
        break
    end
    if (norm(x-b)/normb > 1e5)
        disp('Divergence!');
        break
    end
    
    if EPS
        y1 = max( y1 + delta*( -(x-b) - EPS), 0 );
        y2 = max( y2 + delta*(  (x-b) - EPS), 0 );
        if USE_SLOW_UPDATE
            % mex file not installed, so do this instead:
            Y = updateSparse_slow(Y,y1-y2,indx,indx_i,indx_j);
        else
            updateSparse(Y,y1-y2,indx);
        end
    else
        y = y + delta*(b-x);
        if USE_SLOW_UPDATE
            % mex file not installed, so do this instead:
            Y = updateSparse_slow(Y,y,indx,indx_i,indx_j);
        else
            updateSparse(Y,y,indx);
        end
    end
    MSE(1,k) = norm((M - U*Sigma*V').*Omega,'fro')/norm(M,'fro');
    if (MSE(1,k) < tol)
        break
    end
end

OUT = U * Sigma * V';
MSE_out = MSE(1,k);

if VERBOSE==1, fprintf('\n'); end
numiter = k;
out.residual = out.residual(1:k,:);
out.time = out.time(1:k,:);
out_time = toc;
out.rank= out.rank(1:k,:);
out.nuclearNorm= out.nuclearNorm(1:k,:);