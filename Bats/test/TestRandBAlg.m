%TESTRANDBALG
% Tests the random implementation of BAlg.

clear all; s = RandStream('mcg16807', 'Seed',sum(100*clock)); RandStream.setDefaultStream(s); % clears all variables and reinitialises the random generators

% initialisation of environment
N = 10; % 10 arms
rl = rand(1,N); % rewards list
e = EnvironmentBandit('normal', {rl, 0});


% BASIC TEST
b = RandBAlg(N); % initialises the bandit
for i=1:2 % 2 iterations
	xs = b.choose();
	ys = e.play(xs);
	b.train(xs,ys);
end
% make sure there were 2 iterations
assertEqual(b.t, 2);
% make sure training set is empty
assertEqual(b.ntr, 0);


% TEST CHOOSE ARMS + TRAIN
b = RandBAlg(N);
xs = [b.choose() b.choose() b.choose()];
ys = e.play(xs);
b.train(xs,ys);
assertEqual(b.t, 3);


% TEST COPY CONSTRUCTOR
b2 = copymyobj(b);
xs = b.choose();
ys = e.play(xs);
b.train(xs,ys);
% b and b2 should now be different: b has one more iteration than b2
assertEqual(b.t-b2.t, 1);


% TEST PLAYABLE
b = RandBAlg(N);
b.playable = [1:2];
xs = [b.choose() b.choose() b.choose() b.choose() b.choose()];
assertEqual(max(xs), 2);


% TEST CHOOSESIMULATED
b = RandBAlg(N);
xs = b.chooseSimulated(10);
assertEqual(b.t, 0);
assertEqual(length(xs), 10);
