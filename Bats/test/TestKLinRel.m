% 
%%%

clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

% initialisation of environment
N = 10;
K = eye(N);
signoise = 0.3;
e = EnvironmentBandit('gp', {K, signoise}); % generate from multivariate Gaussian with covariance matrix K, so that our belief (encoded by the GP assumption) is appropriate

% initialisation of algorithm
signoise = 0;
ker{1} = K;
ker{2} = log(signoise);
b = KLinRel(ker); % we are using the correct signoise (in general, we do not know it)


%%%
% test general BAlg: use public methods and make sure there are no errors during execution
%%%
for i=1:N
	x = b.choose();
	% one arm, one reward only
	y = e.play(xs);
	b.train(xs,ys);
end
x = b.choose();
xs = b.chooseSimulated(N); 
assertEqual(b.t, N+1); % make sure there were N iterations
ys = e.play(xs); % several arms and rewards
b.train(xs,ys);
% 
% 
% %%%
% % test ucb: with identity kernel matrix, estsignoise to 0, B(t) = 2.*sqrt(log(t))./signoise, and forcing KLinRel to do the same initialisation as ucb, KLinRel should have almost the same M values as ucb
% % WE NEED TO USE REWARDS CENTRED AROUND 0 FOR BOTH UCB AND KLinRel
% %%%
% 
% % environment
% rl = rand(1,N).*2-1; % rewards list in [-1,1]
% signoise = 0;
% e = EnvironmentBandit('normal', {rl, signoise});
% 
% % KLinRel
% g = KLinRel({eye(N), log(signoise)});
% g.beta = @(t) 2.*sqrt(log(t))./signoise;
% 
% % ucb
% u = ucb(N);
% 
% % initialisation phase
% for i=1:N % as many iterations as there are arms
% 	ys = e.play(i);
% 	g.train(i,ys);
% 	u.train(i,ys);
% end
% assertElementsAlmostEqual(g.M, u.M .* u.nplayed ./ (u.nplayed + signoise));
% 
% 
% 
% %%%
% % other tests
% %%%
% 
% % generate random kernel matrix K
% K = rand(N,N); K = normalise(K*K');
% e = EnvironmentBandit('gp', {K, signoise});
% b = KLinRel({K, log(signoise)});
% 
% % IDEA: see if, with the same training set, KLinRel and linrel have the same M values?
% 
% % After one iteration, we shouldn't have any Inf values in V or U anymore, and all M values should have changed from 0
% x = b.choose();
% y = e.play(x);
% b.train(x,y);
% assertEqual(all(M==zeros(1,N)), false);
% assertEqual(all(V==ones(1,N).*Inf), false);
% 
% % test the getter for M: do we get the values in [0,1] or in [-1,1] ??
% assertTrue(min(M)>=0);
% assertTrue(max(M)<=1);
% 
% % do N iterations and test covariance matrix is correct
% b = e.iterations(b, N);
% assertElementsAlmostEqual(b.Cti, covinv(b.K, signoise));


% Test constructor with labels and features, test addArm
clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);
N = 10;
delta = 0.05;
signoise = 0.1;
sigma = 1;
labels = [];
dim = 5;
xf = ones(dim, 1); % arm to be added again later
features = [xf randn(dim, N-1)]; % create N random vectors of dim dimensions, normally distributed
ker{1} = {'covSum', {'covSEiso','covNoise'}};
ker{2} = [log(sigma); log(sqrt(1)); log(signoise)];
b = KLinRel(ker, labels, features, delta);
e = EnvironmentBandit('gp', {b.K, signoise}); % create environment which rewards centered around 0 and drawn from multivariate gaussian with cov matrix taken from b
x = b.choose();
y = e.play(x);
b.train(x, y);
U = b.estimateU(xf); % predict U for xf and check it corresponds to b.U(1)
assertEqual(U, b.U(1));
b.addArm([], xf); % bandit and environment are not in sync anymore
assertEqual(b.K(:,1), b.K(:,b.N)); % check kernel products: b.K(:,1) == b.K(:,b.N)
% check that M, V and U for first and last arm are the same
assertEqual(b.M(1), b.M(b.N));
assertEqual(b.V(1), b.V(b.N));
assertEqual(b.U(1), b.U(b.N));


% Test with un-normalised kernel

% Test with several sigma noises for each arm

% Test hyperparameter learning

% Unit tests for each method
