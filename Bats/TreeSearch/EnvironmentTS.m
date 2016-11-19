classdef EnvironmentTS < Environment
%ENVIRONMENTTS class for tree search problems:
%   Derives from the Environment class.
%   Gives rewards to sequences of node labels (i.e. paths).
%   Indicates which children nodes exist for a given node, through the offspring function.
%   Assumes the best possible reward is 1.
%
%   EnvironmentTS Properties:
%      R         - list of regret values
%      offspring - lists the children that can be produced from a given node
%
%   See also OFFSPRINGSUM, REWARDSUM (in the test files).
	

	properties (GetAccess='public', SetAccess='private')
		R = []; % list of cumulative regret values; max regret is simply bounded by the last entry divided by number of iterations
		
		offspring;
		%OFFSPRING
		%   Gives features of child nodes available after the last element of a given path.
		%   Used by the tree search algorithm during the exploration of a tree. The nodes it outputs can be stored in a tree structure.
		%
		%   INPUTS
		%      LA  - feature representations of nodes of a given path, so that we know where we come from (size dim x depth-of-last-node)
		%            (note that we can't give a sequence of node indices since these are specific to the tree structure)
		%
		%   OUTPUTS
		%      LC  - list of children feature representations
		
    end

	properties (GetAccess='private', SetAccess='private')
		rbest = 1; % mean reward of the best arm; we don't actually know which is the best arm
		% Tim = zeros(nb,1); % vector of total execution time at each time step
	    % Mem = zeros(nb,1); % vector of total memory consumption at each time step (in bytes)
    end

	% The 'reward' property was already declared in the superclass, but we expand on its expected profile in the comments below:
	%
	%REWARD (private)
	%   Gives the reward for a path, based on the node features vectors.
	%   
	%   INPUTS
	%      LA  - list of node feature vectors for this path (size dim x maximum-depth)
	%
	%   OUTPUTS
	%      r   - reward
	
	methods (Access='public')
		
		% Constructor
		%
		% INPUTS
		%
		% offspring =	function that, given a node's and its ancestors' labels (presented as a cell array of elements in order of depth), outputs the list of labels of this node's children (note that labels can be anything, not necessarily numbers)
		% 				siblings can't have same labels
		%				this function is used to build the tree and is called when an unexplored node is visited for the first time
		% 				function L = offspring(lpath)
		%
		% reward =		function that, given a path's node labels, outputs the reward associated to this path
		%	  			function value = reward(lpath)
		%
		% OUTPUTS
		% - obj: an initialised environment
		
		function obj = EnvironmentTS(reward, offspring, rbest)
			obj.reward = reward;
			obj.offspring = offspring;
			if (nargin>2)
				obj.rbest = rbest;
			end
        end
		
		function [y obj] = play(obj, x) % see superclass
			[y obj] = play@Environment(obj, x'); % we store tree paths as v. vectors!
			if (isempty(obj.R))
				Rnew = obj.rbest - y;
			else
				Rnew = obj.R(end) + (obj.rbest - y);
			end
			obj.R = [obj.R Rnew];
		end
	
	end
	
end