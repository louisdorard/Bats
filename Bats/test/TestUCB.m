%TESTUCB
% Tests the UCB-1 implementation contained in the UCB class.

clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s);

% initialisation of environment
N = 10; % 10 arms
rl = rand(1,N); % rewards list in [0,1]
signoise = 0;
e = EnvironmentBandit('normal', {rl, signoise});


% TEST UCB's INITIALISATION PHASE
b = UCB(N); % initialises the bandit
for i=1:N % as many iterations as there are arms
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
assertEqual(b.t, b.ntr); % t should be equal to ntr
assertEqual(length(unique([e.X{:}])), N); % make sure that each arm has been played once
assertEqual(b.nplayed, ones(1,N)); % same test but using nplayed


% TEST M UPDATES
M = b.M;
% play N more iterations: M values shouldn't change and should be equal to rl
for i=1:N
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
M2 = b.M;
assertElementsAlmostEqual(M, M2);
assertElementsAlmostEqual(M, rl);
assertEqual(all(b.nplayed==2.*ones(1,N)), false); % it would be unlikely that all arms would have been played the same number of times!


% TEST REWARD PLUS NOISE
signoise = 0.3;
e = EnvironmentBandit('normal', {rl, signoise});
% play N iterations
for i=1:N
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
M = b.M;
% play N more: M should change
for i=1:N
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
M2 = b.M;
assertEqual(all(M==M2), false); % M and M2 should be different


% TEST RANDOMISATION OF CHOOSE
N = 7;
x1 = [1; 0; 0];
x2 = [0; 1; 0];
x3 = [0; 0; 1];
e = EnvironmentBandit('normal', {[0 0.5 0 0 0.5 1 1], 0}, x1); % 0 noise so that after initialisation (N iterations) the M estimates converge to the reward list; some arms will have same M values, and same V values after initialisation, so they'll have same U values
b = UCB(N);
for i=1:N
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
xs1 = b.chooseSimulated(N);
xs2 = b.chooseSimulated(N);
assertEqual(all(xs1 == xs2), false); % would be unlikely if they would be the same


% TEST LABELS AND FEATURES
b = UCB(11, e.features);
b.addArm(12, e.addArm(0, x2)); % new arm has reward of 0
b.addArm(13, e.addArm(1, x3)); % " of 1
% features should make identity matrix
assertEqual(b.features, eye(3))
% all arms M values should be 0 and V and U should be infinity
assertEqual(b.M, zeros(1,3));
assertEqual(b.V, Inf .* ones(1,3));
assertEqual(b.U, Inf .* ones(1,3));
b.train(12, 0.5, 'label');
assertEqual(b.nplayed, [0, 1, 0]); % second arm should be in training
