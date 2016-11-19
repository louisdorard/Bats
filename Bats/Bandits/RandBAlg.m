classdef RandBAlg < BAlg
% RANDBALG
%   Random bandit algorithm implementation
%
%   See also BALG.

    methods (Access='public')
        
		function obj = RandBAlg(varargin)
		%RANDBALG
		%   Constructor. See super class. Sets S to 0.
		
			obj = obj@BAlg(varargin{:}); % apply constructor of super class
			obj.S = @(t) 0; % the algorithm is random, so no need to keep training set in memory
		end

		function x = choose(obj, varargin)
		%CHOOSE
		%   Chooses an arm to be played by the environment, randomly.
			
			rd = rand;
			NP = length(obj.playable);
			x = obj.playable(floor(1+rd.*NP));
			if (length(varargin)>=1 && strcmp(varargin{1}, 'label'))
                x = obj.labels(x);
			end
        end
        
    end


end