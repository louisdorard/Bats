function y = bernoulli(p,n); 

% BERNOULLI.M 
% This function generates n independent draws of a Bernoulli 
% random variable with probability of success p. 
% first, draw n uniform random variables 
%
% Author: Keisuke Hirano
% http://www.u.arizona.edu/~hirano/520_2006/matlab_intro.pdf

x = rand(n,1); 

% set y=1 if x is less than p. This gives probability p of success 
y = (x <= p); 

% end function definition 
