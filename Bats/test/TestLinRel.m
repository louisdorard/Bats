%%%
% Test of the LinRel algorithm
%%%

clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

nit = 8;
N = 8;
dim = 10;
signoise = 0.01; % regularisation for KLinRel
features = randn(dim, N); % create N random vectors of dim dimensions, normally distributed
labels = rand(1, N);
delta = 0.05;
e = EnvironmentBandit('normal', {labels, 0}, features);
features = e.features; % they may have changed if we normalise them in the environment
b = LinRel([], features, delta); % initialises the bandit
K = features' * features; % linear kernel
b2 = KLinRel({K, log(signoise)});
b.beta = @(t) 1; b2.beta = @(t) 2;


for i=1:nit
	xs = i;
	ys = e.play(xs);
	b.train(xs,ys);
	b2.train(xs,ys);
end

[l i] = sort(b.U); i
[l i] = sort(b2.U); i
U = TestLinRelOriginal(nit, features, labels);
[l i] = sort(U); i