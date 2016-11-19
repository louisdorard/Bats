t = 10;
sigma = 0.1;

K = rand(t);
K = K*K';
K2 = K(2:t,2:t);

Ctiall = inv(K+sigma.^2.*eye(t));

Cti = subcovinv(Ctiall);
Cti2 = inv(K2+sigma.^2.*eye(t-1));
% Cti should be equal to Cti2

e = K(1,1) - K(2:t,1)'*Cti*K(2:t,1) + sigma.^2;
d = Cti*K(2:t,1);
A = Cti + 1./e.*d*d';
Cti3 = [1./e -d'./e; -d./e A];
% Cti3 should be equal to Ctiall