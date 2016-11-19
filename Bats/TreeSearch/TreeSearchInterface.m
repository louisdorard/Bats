classdef TreeSearchInterface < handle
%TREESEARCHINTERFACE
%   Abstract super class that specifies the profile of the "choose" method
%   that all tree search algorithms should implement.

	methods (Abstract)
		
		[p d] = choose(obj, np, nit)
		%CHOOSE
		%   Choose a near-optimal path of length np to output, after nit iterations
		%
		%   INPUTS
		%      np  - length of path to output
		%      nit - number of iterations allowed
		%
		%   OUTPUTS
		%      p   - features of nodes in chosen path (of size dim x np, where dim is the dimension of the node feature space)
		%      d   - bound on the estimated distance to maximum
		
		% IDEA: we could have, as an alternative stopping criterion, distance to maximum smaller than epsilon; or give a time budget
		
	end

end