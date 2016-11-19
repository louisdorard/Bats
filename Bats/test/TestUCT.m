%clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);


signoise = 0.1;
D = 10;
rootFeature = [0; 1; 0];
B = 5;
o = @(LA)offspringSum(B, LA);
r = @rewardSum;

% determine rbest for these offspring and reward functions
s = RandStream('mcg16807', 'Seed', 1);
prevstream = RandStream.getDefaultStream();
RandStream.setDefaultStream(s);
rbest = max(randn(1,B)).*D;
RandStream.setDefaultStream(prevstream);

nit = 100;

ts = UCT(rootFeature, o, r, D, rbest, 'ucb', 'iterative-deepening');
ts.choose(D, nit);

ts.e.R(end)

% GP_UCT test...
