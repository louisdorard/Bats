clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

signoise = 1;
D = 10;
rootFeature = [0; 1; 0];
B = 5;
gamma = 0.75;
seed = 1;
o = @(LA) offspringSum(seed, B, LA);
r = @(LA) rewardSum(gamma, signoise, LA);

% determine rbest for these offspring and reward functions
s = RandStream('mcg16807', 'Seed', 1);
prevstream = RandStream.getDefaultStream();
RandStream.setDefaultStream(s);
rbest = max(randn(1,B)).*D;
RandStream.setDefaultStream(prevstream);

nit = 100;

ker{1} = 'covPathsDISC';
ker{2} = log(0.5);
% ts = GPTS(rootFeature, o, r, D, rbest, ker, signoise, @(t) t);
% ts.choose(D, nit);

% ts = UCT(rootFeature, o, r, D, rbest, 'random', 'iterative-deepening');
% ts.choose(D, nit);
% 
ts = UCT(rootFeature, o, r, D, rbest, 'ucb', 'depth-first');
ts.choose(D, nit);
