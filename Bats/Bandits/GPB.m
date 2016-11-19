classdef GPB < kRRUcbAlg
% GPB
%   Gaussian Processes Bandit algorithm
%   See Srinivas et al. (2010) for a description of the algorithm.
%   The GPB algorithm assumes a GP reward with zero mean.
%
%   delta represents the regret bound confidence threshold delta
%   (see Srinivas et al. 2010)
%
%   GPB Methods:
%      GPB         - constructor
%      learnHyper  - learns logtheta by maximum likelihood
%
%   See also KUCBALG, GPML/MINIMIZE, GPR2, COVINV, SUBCOVINV.

    methods (Access='public')

		function obj = GPB(varargin)
			obj = obj@kRRUcbAlg(varargin{:}); % apply constructor of super class
        end
		
		function obj = learnHyper(obj)
		%LEARNHYPER
		%   Learns hyperparameters logtheta by maximising the likelihood of observed data.
		%   This is only usable if we know the kernel and its parameters, and if a feature representation of the new arm is given.
		%   Works only for cov functions that are differentiable with respect to the hyperparameters.
		%   Based on the 'minimize' function of the GPML toolbox, and a modified 'gpr' function.
			
			if (isempty(obj.covfunc))
				error('We need to know a feature representation of the new arm and to have defined a covariance function.');
            end
			
            try
				l = length(obj.logtheta);
				fix = l+1; % this corresponds to the noise variance; if we don't fix it, it can become very small and cause numerical instabilities
				if (strcmp(obj.covfunc, 'covSEiso') || strcmp(obj.covfunc, 'covSEard'))
					fix = [l fix]; % we also fix the signal variance
				end
                newlogth = minimize([obj.logtheta; log(obj.signoise)], 'gpr2', -100, {'covSum', {obj.covfunc,'covNoise'}}, obj.features(:,obj.Tr.X)', obj.Tr.Y', [], fix);
                obj.logtheta = newlogth(1:end-1); % this also updates K and A
            catch e
                e % exceptions can be raised due to numerical instabilities
            end
			% gpr2 takes covfunc, x and y (training data) in input. It outputs minus the log likelihood of the data, along with its partial derivatives with respect to the hyperparameters. For this, it uses the partial derivatives given by covfunc.
			% Hence, it is gpr2 that we try to minimize, and the minimize function uses the partial derivatives for that, starting from the previous hyperparameter values.
			% We allow minimize to perform 100 iterations.
			% See http://www.gaussianprocess.org/gpml/code/matlab/doc/regression.html#ard for more info.
            
		end
		
    end
	
	methods (Access='protected')
        
		function V = k2V(obj, k, k1)
			V = k1' - diag(k' * obj.Ci * k)'; % we ditch the components that are not on the diagonal... maybe a for loop would be more efficient if k is a matrix?
		end
		
		function dV = al2dV(obj, al)
			dV = - al.^2./(obj.V(obj.Tr.X(obj.ntr)) + obj.signoise.^2);
		end
		
		function obj = downdateV(obj, x, y, Mprev)
			% compute Vprev; note that the oldest element of the training set hasn't been removed yet
			k = obj.K(obj.Tr.X, x);
			k1 = obj.K(x, x);
			Vnext = obj.k2V(k, k1); % Vnext refers to sigma_{t,l}(x_{t-l})
			
			% compute V
			l = (Mprev-obj.M).*(Vnext+obj.signoise.^2)./(y-obj.M(x));
			obj.V = obj.V + l.^2./(Vnext+obj.signoise.^2);
		end

        function obj = updateBeta(obj)
            obj.beta = @(t) sqrt(2 .* log((obj.N .* t.^2 .* pi.^2) ./ (6.*obj.delta))); % as defined in (Srinivas et al. 2010)
        end
		
    end
end