classdef LinRel < UcbAlg
	

	%%%
	% Attributes
	%%%
    
    
	% Internal parameters for running the algorithm: training set, kernel matrix, posterior mean and variance
	% These can be accessed by the outside, but they are only set
	% internally.
	% - delta: confidence threashold for regret bound, as in (Srinivas et al. 2010), here set to 95%
    % - K: kernel matrix between arms
	% - logtheta: log hyperparameters
	% - estsignoise: what we think the noise variance is
    properties (GetAccess='public', SetAccess='protected')
		A; % used for computing M and V
    end
    
    
	%%%
	% Public methods
	%%%

    methods (Access='public')


		% Constructor
		% INPUTS
		% - delta: confidence interval for regret bound (impacts B)
		% - labels
		% - features
		% OUTPUTS
		% - obj: the GPB object that represents the resulting bandit algorithm
		function obj = LinRel(varargin)
			% apply constructor of super class
            obj = obj@UcbAlg(varargin{:});
        end


		function obj = addArm(obj, varargin)
			
			% this is only usable if a feature representation of the new arm is given
			if (length(varargin)<2)
				error('We need the feature representation of the new arm.');
            end
			
            obj = addArm@UcbAlg(obj, varargin{:});
			
			% now we need to compute M, V and U for the new arm (they've been initialised to 0, Inf, Inf by addArm@UcbAlg, but here we use regression)
            if (obj.ntr>0)
                x = varargin{2}; % feature representation of the new arm
				a = obj.Ci * x;
			    na = sqrt(sum(a.^2,1)); % calculate 2-norm of a
				obj.M(obj.N) = Y*obj.a;
				obj.V(obj.N) = na;
				obj.U(obj.N) = obj.M(obj.N) + obj.beta(obj.t) .* sqrt(obj.V(obj.N));
            end
			
		end

		
		function U = estimateU(obj, x)
			
			% x: test point
            a = obj.Ci * x;
		    na = sqrt(sum(a.^2,1)); % calculate 2-norm of a
			U = Y*a + B(t).*na;
			
		end
		

    end
    

	%%%
	% Private methods used by the above
	%%%

    methods (Access='protected')
		
		function obj = removeOldestTraining(obj)
			if (obj.ntr>1) % we cannot remove an element of the training set if the training set has only one element...
				XF = obj.features(:,obj.Tr.X);
                dim = size(obj.features, 1);
			    obj.Ci = XF'/(XF*XF' + signoise.^2 .* eye(dim));
				obj = removeOldestTraining@BAlg(obj, x, y);
			end
		end
		
		% Update the M and V variables when adding a new arm x with reward y to the training set
		% Precondition: (x,y) hasn't yet been added to obj.Tr
		function obj = updateMV(obj)
			XF = obj.features(:, obj.Tr.X);
			signoise = 0.1; % TODO: parametrise this
            dim = size(obj.features, 1);
		    obj.Ci = XF'/(XF*XF' + signoise.^2 .* eye(dim));
			a = obj.Ci * obj.features;
		    na = sum(a.^2, 1); % square norm of a
			obj.M = obj.Tr.Y * a;
			obj.V = na;
        end

		function obj = updateBeta(obj)
			obj.beta = @(t) sqrt(2 .* log(2.* obj.N .* t ./ obj.delta));
		end
		
    end
end