classdef UCB_bast < UCB

    methods (Access='public')

		function obj = UCB_bast(varargin)
			obj = obj@UCB(varargin{1:end-1});
			obj.delta = varargin{end};
        end

	end
	
    methods (Access='protected')
        
        function obj = updateBeta(obj)
            obj.beta = @(t) sqrt(log(2.*obj.N.*t.*(t+1).*obj.delta^(-1))./(2.*t));
        end
        
    end


end