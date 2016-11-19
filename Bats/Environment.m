classdef Environment < handle
%ENVIRONMENT class for reinforcement learning problems
%   Performs calls to the reward function through the 'play' method, and
%   keeps track of arms played and rewards obtained.
%   
%   Environment Properties:
%      reward        - (Read-only) real-valued reward function
%      t             - (Read-only) number of plays (incremented
%                      as we call the function play)
%      X             - (Read-only) horizontal vector of arms chosen so far
%      Y             - (Read-only) horizontal vector of rewards received so
%                      far
%
%   Environment Methods:
%      play          - plays arms by invoking the reward function and
%                      updating variables X, Y and t accordingly.

	
	properties (GetAccess='public', SetAccess='protected')
		reward; % real-valued function
        t = 0; % number of iterations/plays, incremented as we call the function play
		X = {}; % horizontal vector of arms chosen so far, size 1*t
		Y = []; % horizontal vector of rewards received so far, size 1*t
    end
	
	methods (Access='public')
		
		function [ys obj] = play(obj, xs)
		%PLAY
		%   Plays arms by invoking the reward function and updating variables X, Y and t accordingly.
		%
		%   INPUTS
		%      xs  - list of arm indices to by played
		%
		%   OUTPUTS
		%      ys  - list of obtained rewards
		
			for i=1:length(xs)
				obj.t = obj.t+1;
				obj.X(obj.t) = {xs(i)}; % cell, to allow the fact that arms may have different dimensions, as in tree search
				ys(i) = obj.reward(xs(i));
				obj.Y = [obj.Y ys(i)];
			end
		end
		
		function obj = reset(obj)
			obj.X = {};
			obj.Y = [];
			obj.t = 0;
		end
	
	end
	
end