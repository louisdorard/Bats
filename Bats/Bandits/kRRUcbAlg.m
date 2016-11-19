classdef kRRUcbAlg < UcbAlg
% KUCBALG
%   Abstract super class for ucb-type bandit algorithms that
%   perform kernel Ridge Regression to determine M.
%
%   M and V can be expressed as affine functions of the total
%   kernel matrix and the inverse covariance matrix Ci, determined
%   from the training kernel matrix and of size ntr x ntr.
%   
%   The dM and dV updates can be expressed as simple functions
%   of the alpha values of arms, which are themselves expressed
%   as linear functions of the total kernel matrix and the Q matrix.
%   See notations and formulae in thesis.
%
%
%   kRRUcbAlg Properties:
%      updateMode - (Read-only) indicates which version of the 
%                   algorithm to run (default, online 1 or 2)
%      covfunc    - (Read-only) kernel/covariance function
%      logtheta   - (Read-only) log hyper-parameters
%      signoise   - (Read-only) noise standard deviation in the
%                   GP regression model, which is equivalent to a
%                   regulariser in the kRR model
%      K          - (Read-only) total kernel matrix between arms
%      Ci         - (Read-only) inverse covariance matrix,
%                   used for default and 'online1' updates
%      Q          - (Read-only) matrix of q_t vectors for all arms,
%                   used for 'online2' updates
%
%   kRRUcbAlg Methods:
%      kRRUcbAlg - constructor
%      estimateU - computes the U value for a new arm feature vector
%      addArm    - see super class; the initialisation of M and
%                  V is different here
%
%   See also GPML/COVFUNCTIONS, UCBALG, GPB, KLINREL.


	properties (GetAccess='public', SetAccess='protected')
		updateMode = ''; % 'online1' or 'online2' if we wish to perform online updates to M and V, empty otherwise
		covfunc = ''; % must be a covariance function defined in the GPML toolbox (see GPML/COVFUNCTIONS), but not including noise
		logtheta = []; % for previous covfunc example, we should give as hyperparameters: [log(sigma), log(sqrt(sf2))] (first element is the characteristic length scale; second element is the signal variance and should be 0 for a normalised Gaussian kernel)
		signoise; % noise standard deviation
		K; % total kernel matrix, of size [N, N]
		Ci = []; % this could actually be a private property, but we make it readable for our unit tests
		Q = []; % has size t x N
	end
	
	properties (GetAccess='protected', SetAccess='protected')
		Vprev; % used by updateQ to go from Q_t to Q_{t+1}, and represents sigma_t(x) where x is the last arm added to training
	end

	% Getters and setters
	methods

		function obj = set.logtheta(obj, logtheta)
		% Change the hyperparameters => we must recompute the kernel matrix
	        obj.logtheta = logtheta;
			if (~isempty(obj.covfunc) && ~isempty(obj.features))
				obj.K = obj.kernelProducts(1:obj.N); % this also updates Ci and Q
			end
	    end

		function obj = set.K(obj, K)
		% Change the kernel matrix
			obj.K = K;
	        if (obj.ntr>0 && size(K,1)<=size(obj.K,1)) % if Ci has been defined already (ntr>0) and if we are changing the value of existing elements (new matrix isn't bigger than existing one)
	            
				% recompute Ci and Q from scratch
				obj.Ci = inv(obj.K(obj.Tr.X,obj.Tr.X) + diag(obj.signoise)^2.*eye(obj.ntr)); % update Ci in consequence
				if (strcmp(obj.updateMode, 'online2'))
					obj.Q = obj.Ci * obj.K(obj.Tr.X,:);
				end
				
				% update M, V and U in consequence
				kk = obj.K(obj.Tr.X, :); % matrix of kernel products between all arms and arms in training (which INCLUDES the new arm)
				kk1 = diag(obj.K); % vector of kernel products of each arm with itself
				obj.M = obj.k2M(kk);
				obj.V = obj.k2V(kk, kk1);
				obj = obj.updateU();

	        end
			% we don't need to update Ci when we are adding elements to it
		end

	end

	methods (Access='public')

		function obj = kRRUcbAlg(varargin)
		%KUCBALG
		%   Constructor.
		%
		%   INPUTS
		%      varargin is:
		%         - 1st case: {{covfunc, logtheta}, signoise, labels, features} or {{covfunc, logtheta}, signoise, labels, features, delta, updateMode}
		%           where covfunc must be specified without noise
		%	      - 2nd case: {K, signoise} or {K, signoise, labels} or {K, signoise, labels, delta, updateMode}
			
			% We need to get the arguments to call the super class constructor with
			ker = varargin{1};
	        if (iscell(ker))
				cas = 1; % 1st case
				args{1} = varargin{3}; % labels
                args{2} = varargin{4}; % features
				if (length(varargin)>=5), args{3} = varargin{5}; end % delta
			else
				cas = 2; % 2nd case
				args{1} = size(ker{1}, 1); % N
				if (length(varargin)>=3), args{2} = varargin{3}; end % labels
				if (length(varargin)>=4), args{3} = varargin{4}; end % delta
	        end
			obj = obj@UcbAlg(args{:}); % apply constructor of super class
			
			
			% Set the kernel and hyperparams of the GP model
			% This can only be done after calling the super class constructor, because what follows will implicitly call the setters on K and logtheta (which reference properties that are set by the super class constructor)
			obj.signoise = varargin{2};
			if (cas==1)
				if (length(varargin)>=6), obj.updateMode = varargin{6}; end
				obj.covfunc = ker{1};
				obj.logtheta = ker{2}; % this also sets obj.K
			else
				if (length(varargin)>=5), obj.updateMode = varargin{5}; end
				obj.K = ker; % kernel matrix for this bandit problem
			end
		
		end

		function obj = addArm(obj, varargin)
		%ADDARM
		%   Adds an arm to the bandit problem. See super class.
		%   The initialisation of M and V is different here
		%   
		%   This is only usable if we know the kernel and its parameters,
		%   and if a feature representation of the new arm is given.
		%   Also, we need to be in the default or the 'online1' update modes.

			if (isempty(obj.covfunc) || length(varargin)<2)
				error('We need to know a feature representation of the new arm and to have defined a covariance function.');
			elseif (strcmp(obj.updateMode, 'online2'))
				error('This is not supported in the online2 update mode.');
			end
			
			% add arm (and its feature vector) to set of arms, extend K
			obj = addArm@BAlg(obj, varargin{:}); % we can't reuse addArm@UcbAlg
			[k k1] = obj.kernelProducts(1:obj.N-1, varargin{2});
			obj.K = [obj.K k; k' k1];
        	
			% initialise M and V for the new arm
	        if (obj.ntr>0)
	            k = obj.K(obj.Tr.X, obj.N);
	            obj.M(obj.N) = obj.k2M(k);
	            obj.V(obj.N) = obj.k2V(k, obj.K(obj.N, obj.N));
			else
				obj.V(obj.N) = obj.V(obj.N-1); % still need to expand V (M was already expanded in addArm@BAlg)
	        end
			
			% update beta and therefore U
			obj.updateBeta();
		
		end
	
		function U = estimateU(obj, xf)
		%ESTIMATEU
		%   Computes the U value for a new arm feature vector,
		%   that is not represented in this bandit algorithm's
		%   set of arms.
		%   This is useful for GP_UCT for instance.
		%   Only supported with default and 'online1' update
		%   modes.
		%
		%   INPUTS
		%      xf  - feature vector of the new arm, as given by the environment (the latter decides if features are normalised or not)
		%
		%   OUTPUTS
		%      U   - estimated U value for this xf
		
			if (strcmp(obj.updateMode, 'online2'))
				error('This is not supported in the online2 update mode.');
			end
			
			[k k1] = obj.kernelProducts(obj.Tr.X, xf);
			M = obj.k2M(k);
	        V = obj.k2V(k, k1);
			U = M + obj.beta(obj.t) .* sqrt(V);
		end

	end

	methods(Access='protected')
	
	
		function [k k1] = kernelProducts(obj, range, x)
		%KERNELPRODUCTS
		%   Computes kernel products. This method has two ways of functioning:
		%   - If only the first argument is specified, then we compute a kernel matrix for the arms specified in this argument.
		%   - If the second argument is specified, then we compute kernel products between the second and the first arguments.
		%
		%   INPUTS
		%      range  - range of indices of arms we want to compute kernel products with/between
		%      x      - (optional) feature vector of new arm to compute kernel products with
		%
		%   OUTPUTS
		%      k      - vector of kernel products between the arms which indices are in range, or between these arms and x
		%      k1     - (if x was specified) kernel product of x with itself
			if (nargin>=3)
	        	[k1 k] = feval(obj.covfunc, obj.logtheta, obj.features(:,range)', x');
			else
				k = feval(obj.covfunc, obj.logtheta, obj.features(:,range)');
			end
		end

		function obj = updateMV(obj)
        	% IDEA: update M, V and U only locally?
			if (obj.ntr>=2 && (strcmp(obj.updateMode, 'online1') || strcmp(obj.updateMode, 'online2'))) % we can't perform online updates if ntr==1
				kk = obj.K(obj.Tr.X(1:obj.ntr-1), :); % matrix of kernel products between all arms and arms previously in training
				kk1 = obj.K(obj.Tr.X(obj.ntr), :); % h. vector of kernel products of each arm with the last arm just added to training
				al = obj.k2al(kk, kk1); % we use the existing Q or the updated Ci for computing al
				obj.M = obj.M + obj.al2dM(al);
				obj.Vprev = obj.V(obj.Tr.X(obj.ntr));
				obj.V = obj.V + obj.al2dV(al);
				% update Q and Ci last
				if (strcmp(obj.updateMode, 'online1'))
					obj = updateCi(obj);
				elseif (strcmp(obj.updateMode, 'online2'))
					obj = updateQ(obj, al);
				end
			else
				% we update (compute) Ci first, based on the new training set
				obj = updateCi(obj);
				
				kk = obj.K(obj.Tr.X, :); % matrix of kernel products between all arms and arms in training (which INCLUDES the new arm)
				kk1 = diag(obj.K); % vector of kernel products of each arm with itself
				obj.M = obj.k2M(kk);
				obj.V = obj.k2V(kk, kk1);
				
				if (strcmp(obj.updateMode, 'online2')) % this is where we initialise Q
					obj = updateQ(obj); % no al parameter to give because ntr==1
				end
			end
			if (any(obj.V<0)) % sanity check: variance should always be positive
				warning('Variance should be positive.');
				obj.V = obj.V.*(1-[obj.V<0]); % set all the negative variances to 0
			end
			% IDEA: only update M, V and U for the arms close to x (others won't really be affected)?
	    end
	
		function obj = removeOldestTraining(obj)
			if (strcmp(obj.updateMode, 'online2'))
				error('This is not supported in the online2 update mode.');
			end
			if (obj.ntr>1) % we cannot remove an element of the training set if the training set has only one element...
				obj.downdateCi();
				x = obj.Tr.X(1); y = obj.Tr.Y(1);
				obj = removeOldestTraining@BAlg(obj);
				if (strcmp(obj.updateMode, 'online1'))
					% because we're doing online updates, we need to downdate M and V, otherwise the future values won't make (theoretical) sense
					
					% compute M based on k2M
					kk = obj.K(obj.Tr.X, :);
					Mprev = obj.M; % Mprev refers to mu_{t,l+1} (i.e. the M values before removing data point)
					obj.M = obj.k2M(kk); % M is now mu_{t,l}
					
					obj = obj.downdateV(x, y, Mprev);
					
				end
			end
		end
		
		
		%%%
		% Methods for the default update mode
		%%%
		
		function M = k2M(obj, k)
		%K2M
		%   Computes M value(s) for arm(s) characterised by set(s) of kernel products with the arms in training.
		%   Used in the default update mode, but also in 'online1' for estimateU and addArm.
		%
		%   INPUTS
		%      k  - matrix of kernel products between the n arms we want M values for, and the arms in training: it has size ntr x n
		%
		%   OUTPUTS
		%      M
			M = (k' * (obj.Ci * obj.Tr.Y'))';
		end
		
		function obj = updateCi(obj)
		%UPDATECI
		%   Update Ci when adding a new arm to the training set
		%   Used in default and 'online1' update modes.
			x = obj.Tr.X(obj.ntr); % that's the new arm in training -- since updateCi is called by updateMV and a precondition is that (x,y) has been added to obj.Tr
			if (obj.ntr>=2) % we know that in this case, some obj.Ci has already been defined
				X = obj.Tr.X(1:obj.ntr-1); % arms that were in training before adding x
				% obj.K(obj.Tr.X, x) is the vector of kernel products between the new training point and the others
            	obj.Ci = covinv(obj.K(X, x), obj.signoise, obj.Ci, obj.K(x,x)); % computes new covariance matrix given previous one (using the block inversion lemma)
			else % in this case, we need to compute Ci from scratch and we can't use the recursive formula for M and V
				obj.Ci = covinv(obj.K(x,x), obj.signoise); % computes covariance matrix from scratch; works also for K not normalised!
			end
			
		end
		
		function obj = downdateCi(obj)
		%DOWNDATECI
		%   Downdate Ci when removing first element from training set
		%   Used in default and 'online1' update modes.
			obj.Ci = subcovinv(obj.Ci); % this downdates Ci (subcovinv only works for removing the first — i.e. oldest — element of the training set)
		end
		
		
		%%%
		% Methods for the 'online1' and 'online2' update mode
		%%%
		
		function al = k2al(obj, k, k1)
		%K2AL
		%   Computes alpha vector used to compute dM and dV (see notations in thesis).
		%   Used in the 'online1' and 'online2' update mode.
		%
		%   INPUTS
		%      k  - matrix of kernel products between n given arms and arms previously in training (which EXCLUDES the new arm)
		%      k1 - h. vector of kernel products of each of the n arms with the last arm just added to training
		%
		%   OUTPUTS
		%      al - vector of alpha values for the n given arms
			if (strcmp(obj.updateMode, 'online1'))
				al = k1 - obj.K(obj.Tr.X(1:obj.ntr-1),obj.Tr.X(obj.ntr))'*obj.Ci*k;
			elseif (strcmp(obj.updateMode, 'online2'))
				al = k1 - obj.Q(:, obj.Tr.X(obj.ntr))'*k; % this should be the same as obj.K(:,x)-obj.K(obj.Tr.X(1:end-1),:)' * inv(obj.K(obj.Tr.X(1:end-1), obj.Tr.X(1:end-1)) + diag(obj.signoise)^2.*eye(obj.ntr-1)) * obj.K(obj.Tr.X(1:end-1),x)
			end
		end
		
		function dM = al2dM(obj, al)
		%AL2M
		%   Compute the difference between new M and previous M.
		%   Used both in the 'online1' and 'online2' update modes.
		%
		%   INPUTS
		%      al - vector of alpha values for n arms for which we want the dM values
		%
		%   OUTPUTS
		%      dM - difference between new M values and previous M values for the n given arms
			x = obj.Tr.X(obj.ntr);
			dM = (obj.Tr.Y(obj.ntr) - obj.M(x)) .* (al ./ (obj.V(x) + obj.signoise.^2)); % we don't need to use obj.Vprev since V has't changed yet
		end
		
		function obj = updateQ(obj, al)
		%UPDATEQ
		%   Update Q when adding a new arm to the training set.
		%   Used in the 'online2' update mode.
		%
		%   INPUTS
		%      al - vector of alpha values for all arms
		
			x = obj.Tr.X(obj.ntr);
			if (obj.ntr>=2)
				obj.Q = [obj.Q - repmat(al./(obj.Vprev + obj.signoise.^2),obj.t-1,1) .* repmat(obj.Q(:,x),1,obj.N) ; al./(obj.Vprev + obj.signoise.^2)]; % this causes imprecisions of the order of 10^-17
			else
				obj.Q = obj.K(x,:)./(obj.K(x,x)+obj.signoise.^2);
			end
		end
		

	end

	methods(Abstract, Access='protected')
		
		V = k2V(obj, k, k1);
		%K2V
		%   Computes V value(s) for arm(s) characterised by set(s) of kernel products with the arms in training.
		%   Used in the default update mode, but also in 'online1' for estimateU and addArm.
		%
		%   INPUTS
		%      k  - matrix of kernel products between the n arms we want M values for, and the arms in training: it has size ntr x n
		%      k1 - vector of kernel products of each of the n arms with itself: it has size n x 1
		%
		%   OUTPUTS
		%      V
		
		dV = al2dV(obj, al);
		%AL2DV
		%   Same as above, but for the 'online2' update mode
		
		obj = downdateV(obj, x, y, Mprev)
		%DOWNDATEV
		%   Implements the downdate of V in the 'online1' update mode, assuming that M has already been downdated.
		%
		%   INPUTS
		%      x and y - input and output that have just been removed from training
		%      Mprev   - M values before removing data point
		
	end


end