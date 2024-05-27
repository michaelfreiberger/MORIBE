function [Z,P]= tauchen(N,rho,sigma,lambda)

%column vector of logstate values in state space, N states
logZ0 = -lambda*sqrt(sigma^2/(1-rho^2));
logZN = lambda*sqrt(sigma^2/(1-rho^2));

DlogZ = (logZN - logZ0)/(N-1);
logZ = linspace(logZ0,logZN,N)';
Z = exp(logZ);

P = zeros(N,N); %transition matrix
for jj = 1:N
    P(jj,1) = normcdf( (logZ(1) + DlogZ/2 - rho*logZ(jj)) / sigma,0,1);
    P(jj,N) = 1 - normcdf( (logZ(N) - DlogZ/2 - rho*logZ(jj)) / sigma,0,1);
end

for ii = 2:(N-1)
    for jj = 1:N
        P(jj,ii) = normcdf( (logZ(ii) + DlogZ/2 - rho*logZ(jj)) / sigma,0,1) - ...
                   normcdf( (logZ(ii) - DlogZ/2 - rho*logZ(jj)) / sigma,0,1);
    end
end

end

