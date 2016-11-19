classdef EnvironmentBandit < Environment
%ENVIRONMENTBANDIT Environment for multi-armed bandit problems
%   Defines rewards and regret measures based on a list of mean-reward
%   values. Allows for feature representations of arms.
%   
%   EnvironmentBandit Properties:
%      R                 - (Read-only) cumulative regret at all time steps
%      eR                - (Read-only) empirical cumulative regret at all time steps
%      N                 - (Read-only) total number of arms
%      features          - (Read-only) arm feature representations
%      meanFeat          - (Read-only) mean feature vector
%      stdFeat           - (Read-only) vector of features' standard deviations
%
%   EnvironmentBandit Methods:
%      EnvironmentBandit - constructor
%      play              - plays arms by invoking the reward function and
%                          updating X, Y, t, R and eR accordingly
%      addArm            - adds a new arm with given mean reward
%      iterations        - performs iterations of a given bandit algorithm
%      createBandit      - creates a GPB instance for this environment's arms
%
%   See also ENVIRONMENT, BALG.

	properties (GetAccess='public', SetAccess='private')
		R = 0; % regret, i.e. cumulative pseudo-regret (with reward means instead of actual reward values); max regret is simply bounded by this divided by number of iterations
		eR = 0; % "true", empirical cumulative regret
		N = 0; % number of arms
		features = []; % arm feature representations (optional)
		meanFeat = []; % mean feature vector from un-normalised
        stdFeat = []; % std feature vector from un-normalised
    end

	properties (GetAccess='private', SetAccess='private')
		rewardList = []; % list of mean rewards
		best = 1; % best arm so that reward(best) is the highest on average
		rbest = 1; % mean reward of the best arm
		rewardType;
    end
	
	methods (Access='public')
		
		function obj = EnvironmentBandit(rewardType, params, features, normalise)
		%ENVIRONMENTBANDIT
		%   Constructor that initialises the arms' feature vectors,
		%   normalises them if specified, and defines a reward function based on its type
		%   and parameters.
		%
		%   INPUTS
		%      rewardType  - 'normal', 'bernoulli' or 'gp'
		%      params      - parameters of the reward function
		%      features    - arm feature vectors
		%      normalise   - boolean: specified whether to normalise
		%                    features (true) or not (false)
		%
		%   OUTPUTS
		%      obj         - a new environment
			
			% get number of arms
			if (strcmp(rewardType,'normal') || strcmp(rewardType,'bernoulli') || strcmp(rewardType,'bernoulli2'))
				obj.N = length(params{1});
			elseif (strcmp(rewardType,'gp'))
				obj.N = size(features, 2);
			end
			
			% get features and normalise, if specified
			if (nargin>=3) % if feature vectors are given
				if (nargin<4)
					normalise = false;
				end
				obj.features = features;
				dim = size(obj.features,1);
	            if (obj.N>1 && normalise) % if we have more than one arm and specify we should normalise the feature vectors
	                obj.normalTrain(); % centre and normalise arm features
	            else
	                obj.meanFeat = zeros(dim,1);
	                obj.stdFeat = ones(dim,1);
	            end
			end
			
			obj.rewardType = rewardType; % we want to remember this for use in the addArm method
			
			% define rewardList
			if (strcmp(rewardType,'normal') || strcmp(rewardType,'bernoulli') || strcmp(rewardType,'bernoulli2'))
				obj.rewardList = params{1};
			elseif (strcmp(rewardType,'gp'))
				ker = params{1};
				K = feval(ker{1}, ker{2}, obj.features'); % compute covariance matrix, given the covariance function (ker{1}), its parameters (ker{2}), and the feature vectors
                obj.rewardList = mgd(1, obj.N, zeros(1,obj.N), K); % generate a N-dimensional sample from a Gaussian with zero mean and covariance matrix params{1}
			end
			
			% define reward function
			if (strcmp(rewardType,'normal') || strcmp(rewardType,'gp'))
				if (length(params{2})==1)
					signoise = params{2} .* ones(1,obj.N);
				else
					signoise = params{2};
				end
				obj.reward = @(i) obj.rewardList(i) + randn().*signoise(i);
			elseif (strcmp(rewardType,'bernoulli'))
				% rewardList is the list of Bernoulli parameters: must be in [0, 1]
				obj.reward = @(i) bernoulli(obj.rewardList(i),1); % 1 bernoulli sample from distribution with mean obj.rewardList(i)
			elseif (strcmp(rewardType,'bernoulli2'))
				% same as above but reward gives -1 instead of 0
				obj.reward = @(i) bernoulli(obj.rewardList(i),1).*2-1;
			else
				error('Reward type not supported');
			end
			
			[discard, obj.best] = max(obj.rewardList); % initialisation of best
			obj.rbest = obj.rewardList(obj.best);
        end
		
		function [ys obj] = play(obj, xs)
		%PLAY
		%   See super class. This also updates R and eR.
		
			[ys obj] = play@Environment(obj, xs);
			Rnew = obj.R(end) + (obj.rbest - obj.rewardList(xs));
			eRnew = obj.eR(end) + (obj.reward(obj.best) - ys);
			obj.R = [obj.R Rnew];
			obj.eR = [obj.eR eRnew];
		end
		
		function [feature obj] = addArm(obj, varargin)
		%ADDARM
		%   Add an arm: this is useful for product recommendation for
		%   instance, or for tree search
		%
		%   INPUTS
		%      varargin is {feature representation of this new arm} if the rewardType is 'gp'
		%                  {mean-reward value for this new arm, feature representation} otherwise
		%
		%   OUTPUTS
		%      feature     - normalised feature representation of the new arm
			
			if (length(varargin)>=2 && ~isempty(varargin{2}))
				feature = obj.normalTest(varargin{2});
				obj.rewardList(obj.N+1) = varargin{1};
			else
				if (strcmp(obj.rewardType,'gp'))
					feature = obj.normalTest(varargin{1});
					obj.rewardList(obj.N+1) = 0; % this should actually be determined from a conditional Gaussian, based on the covariances with already existing arms
				else
					feature = [];
					obj.rewardList(obj.N+1) = varargin{1};
				end
			end
			obj.N = obj.N+1;
			obj.features = [obj.features feature];
			
		end
		
		function b = iterations(obj, b, N)
		%ITERATIONS
		%   Performs a given number of iterations, with the current environment, of a given bandit algorithm
		%
		%   INPUTS
		%      b     - bandit algorithm used to select arms to play
		%      N    - number of iterations to perform
		%
		%   OUTPUTS
		%      b     - updated bandit algorithm
			
			for i=1:N
				x = b.choose();
				y = obj.play(x);
				b.train(x,y);
			end
		end
		
		function b = createBandit(obj)
		%CREATEBANDIT
		%   Creates a GPB instance for the arms in this environment
		%
		%   OUTPUTS
		%      b     - GPB instance initialised with a SE-ARD covariance
		%              and this environment's arms' feature vectors

			ker{1} = 'covSEard'; % kernel/covariance function to be used
			dim = size(obj.features, 1);
			ker{2} = [zeros(dim,1); 0]; % hyperparameters: all weights initalised to 1; signal variance initialised to 1
			ker{3} = 0.1; % signoise
			delta = 0.05;
			b = GPB(ker, [], obj.features, delta); % create Bandit Algorithm
		end
	
	end
	
	methods (Access='protected')
		
		function obj = normalTrain(obj)
		%NORMALTRAIN
		%   Normalises feature set that has been given to the constructor
		
			f = obj.features;
            obj.meanFeat = mean(f')';
            obj.stdFeat = std(f')';
			m = repmat(obj.meanFeat,1,size(f,2)); % vector
			f = f - m;
			s = repmat(obj.stdFeat,1,size(f,2)); % vector
			obj.features = f./s; % f should be a sequence of vectors such that the mean value of the ith feature is 0 and the std is 1, for all i
        end
        
        function f2 = normalTest(obj, f)
		%NORMALTEST
		%   Normalises feature vector of a new arm
		
            if (~isempty(obj.meanFeat) && ~isempty(obj.stdFeat))
                f2 = (f - obj.meanFeat)./obj.stdFeat;
            else
                f2 = f;
            end
        end
		
	end
	
end