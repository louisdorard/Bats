classdef BAlg < handle
%BALG abstract super class for bandit algorithms
%   Defines a basic interface and implements some methods shared by all
%   bandit algorithms.
%   
%   BAlg Properties:
%      S               - growth rate of the size of the training set,
%                        specifying when training data should be discarded (when ntr > S(t))
%      playable        - indices of arms that can be played
%      chooseNew       - true if we want to only choose arms never chosen before
%      N              - (Read-only) number of arms
%      t               - (Read-only) number of iterations that have been performed
%	                     = current iteration/time-step minus 1
%                        = ntr for S(t) = t (default)
%                        should match the t value of the environment
%      M               - (Read-only) estimated rewards for all arms = mu_t
%      labels          - (Read-only) list of labels to identify arms
%      dim             - (Read-only) dimension of the feature vectors
%      features        - (Read-only) array of arm feature vectors
%      Tr              - (Read-only) training set
%      ntr             - (Read-only) number of elements in the training set (<= S(t))
%
%   BAlg Methods:
%      BAlg            - initialises the bandit algorithm
%      addArm          - adds an arm to the bandit problem (to be called after
%                        the arm has been added to the environment)
%      choose          - (Abstract) chooses an arm to be played by the environment
%      chooseSimulated - iteratively chooses an arm, plays it, simulates reward,
%                        for as many times as specified
%      train           - adds observed arm-reward pairs to the training set
%
%   See also TRAININGSET, ENVIRONMENTBANDIT.


	% Parameters of how we run the algorithm: can be accessed and set by
	% the outside (can even be changed after a few iterations).
    properties (GetAccess='public', SetAccess='public')
		S = @(t) t; % real-valued, positive and increasing function, <= t
		playable = []; % horizontal vector of size [1, <=N] of integers in [1, N]
		chooseNew = false; % true if we want to only choose arms never chosen before
    end
    
	% Internal parameters for running the algorithm (e.g. training set)
	% These can be accessed by the outside, but they are only set
	% internally.
    properties (GetAccess='public', SetAccess='protected')
		N = 0; % integer
		t = 0; % integer
        M; % h. vector
		labels = []; % list of labels to identify arms
		dim = 0; % dimension of feature vectors
		features = []; % array of arm feature vectors (has to be of size dim x N)
		Tr; % training set: Tr.X will be a horizontal vector of arm indices in training, and Tr.Y a h. vector of rewards for these arms ; this could actually be a private property, but we make it readable for our unit tests
		ntr = 0; % incremented and decremented when necessary, so that it is always equal to Tr.n (and smaller than S(t))
	end

    methods (Abstract, Access='public')
        
		x = choose(obj, varargin) % chooses an arm to be played by the environment, among those that are playable
        
    end

	methods (Access='public')
		
		function obj = BAlg(varargin)
		%BALG
		%   Initialises the bandit algorithm
		%
		%   INPUTS
		%      varargin is either {N} or {N, labels} or {labels, features}
        %      * N is a scalar
        %      * labels is a list of scalars (hence size(labels,1)==1)
        %      * features is a list of (vertical) vectors with at least two feature space dimensions (hence a matrix with size(features,1)>1)
		
            if (length(varargin)>=2 && ~isempty(varargin{2}))
                if (size(varargin{2},1)==1) % this identifies the first case: {N, labels}
                    obj.N = varargin{1};
                    obj.labels = varargin{2};
                else % second case: {labels, features}
                    obj.N = max(size(varargin{1}, 2), size(varargin{2}, 2));
                    obj.labels = varargin{1};
					obj.features = varargin{2};
                end
            else
                obj.N = varargin{1};
            end
			checkUniqueLabels(obj);
            obj.playable = 1:obj.N;
            obj.M = zeros(1,obj.N);
            obj.Tr = TrainingSet();
            obj.ntr = 0;
        end
		
		function obj = addArm(obj, varargin)
		%ADDARM
		%   Adds an arm to the bandit problem.
		%   This must be called after arms have been added to the
		%   environment. The feature vector is given by the environment
		%   and is potentially normalised.
		%
		%   INPUTS
		%      varargin is either {} or {label} or {label, feature}
			
			if (~isempty(varargin))
				label = varargin{1};
				if (length(varargin)>=2)
					feature = varargin{2};
				else
					feature = [];
				end
			else
				label = [];
				feature = [];
			end
			obj.N = obj.N+1;
			obj.playable = [obj.playable obj.N];
			obj.M(obj.N) = 0;
			obj.labels = [obj.labels label];
			obj.features = [obj.features feature];
			
			checkUniqueLabels(obj);
			
		end
		
		
		function x = chooseSimulated(obj, na)
		%CHOOSESIMULATED
		%   Iteratively chooses an arm, plays it, simulates reward given previous learnings on rewards and learns from simulation; this na times and in a copied instance of obj (obj2).
		%
		%   INPUTS
		%      na: number of arms to choose
		%
		%   OUTPUTS
		%      x: h. vector of the indices of the arms with highest ucb value at each iteration (na)
			if (na==1)
				x = obj.choose();
			else
				x = zeros(1,na);
				obj2 = copymyobj(obj);
				for i=1:na
					x2 = obj2.choose();
                    obj2.playable = setdiff(obj2.playable, x2);
					x(i) = x2;
					obj2.train(x2, obj2.M(x2)); % learn from simulated reward
				end
			end
		end
		
		function obj = train(obj, xs, ys, id)
		%TRAIN
		%   Adds arms and observed rewards to the bandit algorithm's training set
		%
		%   INPUTS
		%      xs - h. vector of arm indices
		%      ys - h. vector of rewards for these arms, each in [0;1]
		
			if ~(exist('id', 'var'))
				id = 'index'; % identification type
			end
			for i=1:length(xs)
				% do we have to remove a point from the training data
				% (which size is controlled by S)?
				if (obj.ntr>floor(obj.S(obj.t)))
                    obj.removeOldestTraining();
					% we don't recompute M, this will happen in addTraining
                end
				if (strcmp(id, 'label'))
					armIndex = find(xs(i)==obj.labels); % find the index of the arm that has xs(i) as label
				elseif (strcmp(id, 'feature'))
					armIndex = find(xs(i)==obj.features); % find the index of the arm that has xs(i) as label
				else
					armIndex = xs(i);
				end
				if (isempty(armIndex))
					error('No corresponding arm.');
                end
				obj.t = obj.t + 1;
				obj.addTraining(armIndex, ys(i)); % can only add one element at a time
				if (obj.chooseNew)
					obj.playable = setdiff(obj.playable, armIndex);
				end
			end
        end

		function obj = reset(obj)
			obj.t = 0; % integer
			obj.playable = 1:obj.N;
            obj.M = zeros(1,obj.N);
            obj.Tr = TrainingSet();
            obj.ntr = 0;
		end
		
    end

	methods (Access='protected')
		
		function obj = addTraining(obj, x, y)
		%ADDTRAINING
		%   Adds a (x, y) couple to the bandit algorithm's training set
		%   x - an arm index
		%   y - a reward value
			obj.Tr.add(x, y); % can only add one element at a time
			obj.ntr = obj.ntr+1;
		end
		
		function obj = removeOldestTraining(obj)
		%REMOVEOLDESTTRAINING
		%   Removes the oldest point from the training set
			obj.Tr.removeOldest();
			obj.ntr = obj.ntr-1;
        end
		
		function checkUniqueLabels(obj)
			% checking that no two arms have the same labels
			if (~isempty(obj.labels))
				if (length(unique(obj.labels))<obj.N)
					error('All arms should have different labels');
				end
			end
		end
		
	end

end