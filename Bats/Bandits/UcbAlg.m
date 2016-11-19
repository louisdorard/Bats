classdef UcbAlg < BAlg
% UCBALG
%   Abstract super class for upper confidence bound-type bandit algorithms
%   Implements methods shared by all these algorithms (such as UCB-1, LinRel, etc.)
%   and specifies which methods need to be implemented when creating a new algorithm.
%
%   UcbAlg Properties:
%      V         - (Read-only) list of exploration terms for each arm
%      beta      - (Read-only) balance between M (exploration) and V (exploitation)
%                  controls the width of the confidence intervals
%                  must be positive for t>=1
%      delta     - (Read-only) parameter of beta
%      U         - (Read-only) list of upper confidence bounds for each arm
%
%   UcbAlg Methods:
%      UcbAlg    - constructor: sets beta and initialises V and U
%      addArm    - adds an arm to the bandit problem (see super class)
%      choose    - picks the arm with highest U value
%
%   See also BALG, UCB.
	
    properties (GetAccess='public', SetAccess='protected')
		beta; % real-valued function
        delta; % optional parameter to the beta function
		V = []; % h. vector of uncertainty measures
		U = []; % h. vector of upper confidence bounds
    end
    
    methods
        
        function obj = set.beta(obj, beta)
		% Set the beta function: implicitly called by updateBeta.
	        obj.beta = beta;
			obj = updateU(obj); % if beta has changed, we should update U
	    end
        
    end

    methods (Access='public')
		
		function obj = setBeta(obj, beta)
		%SETBETA
		%   Reserved to advanced users, the form of beta is usually fixed for a given algorithm.
		obj.beta = beta;
		end
		
		function obj = UcbAlg(varargin)
		%UCBALG
		% Constructor (see super class). Sets beta based on given delta, initialises V and U.
		%
		%   INPUTS
		%      varargin{1:2} is either {N} or {N, labels} or {labels, features}
		%      varargin{3} is delta
		
			if (length(varargin)<2), varargin{2} = []; end
			obj = obj@BAlg(varargin{1:2}); % apply constructor of super class
			if (length(varargin)>=3)
                obj.delta = varargin{3};
            else
                obj.delta = 0.05;
			end
			obj.V = ones(1,obj.N).*Inf;
            obj.updateBeta(); % this actually defines beta once delta has been set, and updates U based on the new beta
		end
		
		function obj = addArm(obj, varargin)
		%ADDARM
		%   Adds an arm to the bandit problem (see super class).
		%   Updates the properties that are specific to this class (B, V and U).
		
			obj = addArm@BAlg(obj, varargin{:});
			% update the arm properties that are specific to this class:
			obj.V(obj.N) = Inf;
			obj.U(obj.N) = Inf;
            obj.updateBeta(); % beta might depend on obj.N which has just been incremented
			% the line above implies a call to set.beta, which also updates U
		end

		function x = choose(obj, varargin)
		%CHOOSE
		%   Chooses an arm to be played by the environment, among those that are playable.
		%   Picks the arm with highest U value (and breaks ties arbitrarily).
		%
		%   INPUTS
		%      varargin{1} - (optional) identification mode to use for the output: 'index' or 'label'
		%
		%   OUTPUTS
		%      x           - h. vector of arm ids (index or label)
		
			NP = length(obj.playable);
            if (NP==0)
                x = 0;
            else
                if (obj.ntr==0)
                    rd = rand;
                    x = obj.playable(floor(1+rd.*(NP-1)));
                else
					[discard i] = max(obj.U(obj.playable));
					x = obj.playable(i);
                end
                if (length(varargin)>=1 && strcmp(varargin{1}, 'label'))
                    x = obj.labels(x);
                end
            end

		end
		
		function obj = reset(obj)
			obj = reset@BAlg(obj);
			obj.V = ones(1,obj.N).*Inf;
			obj.U = ones(1,obj.N).*Inf;
		end
		
	end

	methods (Abstract, Access='protected')
		
		obj = updateMV(obj, x, y);
        %UPDATEMV
		%   Updates M and V when adding arm x and observed reward y to the training set.
		%   Precondition: (x,y) has been added to the training set already.
		%   Note that a priori we do not need to downdate M and V when we remove elements of the training set, because they will be recomputed next time we add an arm.
		
        obj = updateBeta(obj);
		%UPDATEB
		%   Defines/updates B based on delta and N
		
	end

    methods (Access='protected')
		
		function obj = addTraining(obj, x, y)
		%ADDTRAINING
		%   See super class. We also update the M, V and U variables.

			% Add (x,y) to the training set
			obj = addTraining@BAlg(obj, x, y);

			% Update the M, V and U variables
			% (do we need to update M, V and U for unplayable arms?)
			obj = updateMV(obj);
            obj = updateU(obj);
			
        end

		function obj = updateU(obj)
			if (obj.beta(obj.t)>0) % if beta(t) is strictly positive, then we can multiply it to V even if the latter is equal to +Inf
				obj.U = obj.M + obj.beta(obj.t) .* sqrt(obj.V); % M, V and U are estimated after t iterations (note that t was incremented before calling addTraining)
			elseif (obj.t==0 || obj.beta(obj.t)==0)
				for i=1:obj.N
                    if (obj.V(i)==Inf)
                        obj.U(i) = Inf;
                    else
                        obj.U(i) = obj.M(i);
                    end
                end
			else
				error('B(t) cannot be negative for t>=1');
			end
		end

    end
	
	
end