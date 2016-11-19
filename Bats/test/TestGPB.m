%TESTGPB
% Tests the GPB implementation of the kRRUcbAlg class

clear; s = RandStream('mcg16807', 'Seed', sum(100*clock)); RandStream.setDefaultStream(s);

% Test constructor when the kernel matrix is given
N = 10;
sigma = 1;
ker{1} = 'covSEiso';
ker{2} = [log(sigma); log(sqrt(1))];
signoise = 0.5;
dim = 5;
xf = ones(dim, 1); % arm to be added again later
features = [xf randn(dim, N-1)]; % create N random vectors of dim dimensions, normally distributed
e = EnvironmentBandit('gp', {ker, signoise}, features); % list of mean rewards generated from multivariate Gaussian with covariance matrix computed from features and given covariance functions and hyperparameters

labels = [];
delta = 0.05;
b = GPB(ker, signoise, labels, e.features, delta); % our belief is appropriate
b1 = GPB(ker, signoise, labels, e.features, delta, 'online1');
b2 = GPB(ker, signoise, labels, e.features, delta, 'online2');


% Test chooseSimulated
b.chooseSimulated(N); 


% Make sure we get the same M, V and U values with all versions of the algorithm

for i=1:(10*N)
	x = b.choose();
	y = e.play(x);
	b.train(x,y);
	b1.train(x,y);
	b2.train(x,y);
end


% Remark on the precision of Matlab's computations
%%%
%
% in the following, we compare values given by different methods and that are supposed to be equal
% this is not the case in practise
% indeed, Matlab is imprecise when working with large vectors
% for instance, if x, y and z are vectors, [x';y']*z is not exactly equal to [x'*z;y'*z]
%
% these imprecisions are amplified with the online methods and after a large number of iterations, as previous computations are reused

assertTrue(norm(b.M - b1.M) < 2.*10.*N.*10.^(-14)); % online updates are slightly imprecise
assertTrue(norm(b.M - b2.M) < 2.*10.*N.*10.^(-14));
assertTrue(norm(b.V - b1.V) < 2.*10.*N.*10.^(-14));
assertTrue(norm(b.V - b2.V) < 2.*10.*N.*10.^(-14));


% Test estimateU for default and 'online1' algorithms

xfn = e.addArm(xf);
U = b.estimateU(xfn); % predict U for xf and check it corresponds to b.U(1)
assertTrue(norm(U - b.U(1)) < 2.*10.*N.*10.^(-15));
assertTrue(norm(U - b1.U(1)) < 2.*10.*N.*10.^(-15));


% Test hyperparameter learning

b.learnHyper();
b1.learnHyper();
b2.learnHyper();

xs = b.chooseSimulated(N);
ys = e.play(xs);
b.train(xs,ys);
b1.train(xs,ys);
b2.train(xs,ys);

assertTrue(norm(b.M - b1.M) < 2.*N.*10.^(-13)); % online updates are slightly imprecise
assertTrue(norm(b.M - b2.M) < 2.*N.*10.^(-13));
assertTrue(norm(b.V - b1.V) < 2.*N.*10.^(-13));
assertTrue(norm(b.V - b2.V) < 2.*N.*10.^(-13));


% Test updateCi and downdateCi for default and 'online1' algorithms

b.S = @(t) N./2;
b1.S = @(t) N./2;
xs = b.chooseSimulated(N);
ys = e.play(xs);
b.train(xs,ys); % because S(t)=N/2, we know that updateCi and downdateCi will have been called for computing A, the inverse covariance matrix
b1.train(xs,ys);
assertTrue(norm(b.Ci - inv(b.K(b.Tr.X, b.Tr.X) + diag(signoise)^2.*eye(b.ntr))) < 11.*N.*10.^(-13));
assertEqual(b.Ci, b1.Ci);
assertTrue(norm(b.M - b1.M) < N.*10.^(-14));


% Test adding arm

b.addArm([], xfn);
assertEqual(b.K(:,1), b.K(:,b.N)); % check kernel products: b.K(:,1) == b.K(:,b.N)
% check that M, V and U for first and last arm are the same
assertAlmostEqual(b.M(1), b.M(b.N));
assertAlmostEqual(b.V(1), b.V(b.N));
assertAlmostEqual(b.U(1), b.U(b.N));

b1.addArm([], xfn);
assertEqual(b1.K(:,1), b1.K(:,b.N)); % check kernel products: b.K(:,1) == b.K(:,b.N)
% check that M, V and U for first and last arm are the same
assertTrue(norm(b1.M(1) - b1.M(b.N)) < 11.*N.*10.^(-7));
assertTrue(norm(b1.V(1) - b1.V(b.N)) < 11.*N.*10.^(-5));
assertTrue(norm(b1.U(1) - b1.U(b.N)) < 11.*N.*10.^(-4));


% Test similarity with UCB-1: with identity kernel matrix and signoise close to 0, GPB should have almost the same M values as UCB-1 if they both have the same training set

% environment
rl = rand(1,N); % in [0,1]
e = EnvironmentBandit('bernoulli', {rl});

% GPB
signoise = 0.001;
g = GPB(eye(N), signoise);
%g.setBeta(@(t) 2.*sqrt(log(t))./signoise);

% ucb
u = UCB(N);
e.iterations(u, N);
g.train(u.Tr.X, u.Tr.Y.*2-1); % rewards are -1 or 1, instead of 0 or 1
assertTrue(norm(g.M-(u.M.*2-1)) < 10.*N.*10.^(-3));
assertTrue(norm(g.V-signoise.^2.*u.V) < 10.*N.*10.^(-3));



% %%%
% % other tests
% %%%
% 
% % generate random kernel matrix K
% K = rand(N,N); K = normalise(K*K');
% e = EnvironmentBandit('gp', {K, signoise});
% b = GPB({K, signoise});
% 
% % IDEA: see if, with the same training set, GPB and linrel have the same M values?
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


% Test with un-normalised kernel

% Test with several sigma noises for each arm
