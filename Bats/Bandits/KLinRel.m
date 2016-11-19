%%%
% Kernel LinRel
%%%

classdef KLinRel < kRRUcbAlg
    
    % When implementing a kRRUcbAlg algorithm, we need to define
    % 1. a beta function parametrised by delta, defined in updateBeta() which is called whenever obj.N or obj.delta is set
    %       -> for KLinRel, delta represents the regret bound confidence threshold delta?
    % 2. k2M and k2V functions as linear functions of a kernel vector k and an intermediate matrix A
    % 3. functions to update and downdate A
    %       -> for KLinRel, ...
    
	methods (Access='public')

		function obj = KLinRel(varargin)
			obj = obj@kRRUcbAlg(varargin{:}); % apply constructor of super class
		end

	end
    
    methods (Access='protected')
		
		function M = k2M(obj, k)
			M = (k' * obj.Ci * obj.Tr.Y')';
		end
		
		function V = k2V(obj, k, k1)
			a = k' * obj.Ci; % horizontal vector
			V = sum(a'.^2,1);
		end
        
        function obj = updateBeta(obj)
            obj.beta = @(t) obj.delta;
        end
		
    end
end