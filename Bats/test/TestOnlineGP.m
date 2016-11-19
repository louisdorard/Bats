clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

n = 5; % number of training points
d = 3; % dimensions
X = rand(d, n);
K = X' * X; % linear kernel matrix
y = randn(n,1); % observed outputs
s = 0.2; % noise standard deviation


for i=1:n
	ssqref(1,i) = K(i,i) - K(1,i).^2 ./ (K(1,1) + s^2);
	muref(1,i) = K(i,1) .* y(1) ./ (K(1,1) + s^2);
end
mu(1,:) = muref(1,:);
ssq(1,:) = ssqref(1,:);
for t=1:n-1
	for i=1:n
		muref(t+1,i) = K(1:t+1,i)' * inv(K(1:t+1,1:t+1) + s^2*eye(t+1)) * y(1:t+1);
		ssqref(t+1,i) = K(i,i) - K(1:t+1,i)' * inv(K(1:t+1,1:t+1) + s^2*eye(t+1)) * K(1:t+1,i);
		mu(t+1,i) = mu(t,i) + (y(t+1) - mu(t,t+1)) * (K(t+1,i) - K(1:t,i)' * inv(K(1:t,1:t) + s^2*eye(t)) * K(1:t,t+1)) ./ (ssq(t,t+1) + s^2);
		ssq(t+1,i) = ssq(t,i) - (K(t+1,i) - K(1:t,i)' * inv(K(1:t,1:t) + s^2*eye(t)) * K(1:t,t+1))^2 ./ (ssq(t,t+1) + s^2);
	end
end

mu
muref
ssq
ssqref