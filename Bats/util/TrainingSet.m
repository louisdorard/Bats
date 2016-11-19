classdef TrainingSet < handle
%TRAININGSET class for machine learning algorithms
%
%   TrainingSet Properties:
%      X            - (Read-only) inputs in training
%      Y            - (Read-only) observed outputs for these inputs
%      n            - (Read-only) number of data points in training
%
%   TrainingSet Methods:
%      add          - add a new data point to the training set
%      removeOldest - remove the oldest data point from the training set

	properties (GetAccess='public', SetAccess='protected')
		X = []; % horizontal vector of inputs in training
		Y = []; % h. vector of observed outputs for these inputs
		n = 0; % integer: number of data points (input-output pairs) in training
	end
	
	methods (Access='public')
		
		function obj = TrainingSet()
		%TRAININGSET
		%   Constructor.
		%   Nothing to do here (see initialisation of properties above).
		end
		
        function obj = removeOldest(obj)
		%REMOVEOLDEST
        %   Remove the first (= oldest) element.
            obj.X = obj.X(2:obj.n);
            obj.Y = obj.Y(2:obj.n);
			obj.n = obj.n - 1;
        end
        
        function obj = add(obj,x,y)
		%ADD
		%   Add input x and observed output y to the training set.
            obj.X(obj.n+1) = x;
            obj.Y(obj.n+1) = y;
			obj.n = obj.n + 1;
        end
		
	end


end