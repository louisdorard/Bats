classdef UCB < UcbAlg
%UCB
%   UCB-1 for multi-armed bandits (Auer et al. 2002)
%   Rewards are assumed to be in [0,1]
%
%   UCB Properties:
%      nplayed   - (Read-only) number of times each arm has been played
%
%   UCB Methods:
%      UCB       - constructor (see super class), initialises nplayed
%      addArm    - see super class
%      choose    - see super class
%
%   See also UCBALG.

	properties (GetAccess='public', SetAccess='private')
		nplayed = []; % lists the number of times each arm has been played (used to define M and V)
    end

    methods (Access='public')

		function obj = UCB(varargin)
			obj = obj@UcbAlg(varargin{:}); % apply constructor of super class
			obj.nplayed = zeros(1,obj.N);
        end

		function obj = addArm(obj, varargin)
			obj = addArm@UcbAlg(obj, varargin{:});
			obj.nplayed = [obj.nplayed 0];
		end
        
    end

    methods (Access='protected')

		function obj = updateMV(obj)
            x = obj.Tr.X(obj.ntr);
			y = obj.Tr.Y(obj.ntr);
			obj.M(x) = obj.nplayed(x)./(obj.nplayed(x)+1) .* obj.M(x) + y./(obj.nplayed(x)+1); % online update of the average (obj.M(x) was initialised to 0)
			obj.nplayed(x) = obj.nplayed(x)+1;
			obj.V(x) = 1./obj.nplayed(x); % see UCB-1 formula
        end
        
        function obj = updateBeta(obj)
            obj.beta = @(t) sqrt(2.*log(t)); % see UCB-1 confidence term definition
        end
        
    end


end