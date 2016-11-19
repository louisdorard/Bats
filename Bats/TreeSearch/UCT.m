classdef UCT < ManyBanditsTS
%UCT algorithm that can be extended to also run BAST and HOO
%
%   Redefines the 'tree' property as a BTree which bandits type (bType) can be any algorithm that doesn't consider arms' feature representations (i.e. the node feature vectors): 'random', 'ucb' or 'bast'.
%   When the parent has a dummy node, the 'next' method creates and plays a child node, randomly chosen among those specified by the environment's 'offspring' function.
%   If we would be using GPB instances in the BTree, the child to be created wouldn't be chosen randomly but based on its predicted U value, determined from its feature vector.
%
%   See also MANYBANDITSTS, BTREE.

	methods

		function obj = UCT(rootFeature, offspring, reward, maxDepth, rbest, bType, growMethod, varargin)
			% constructor
			% varargin represents the extra parameters to be given to BTree: rho, delta, paramA
			obj = obj@ManyBanditsTS(rootFeature, offspring, reward, maxDepth, rbest, growMethod);
			obj.tree = BTree(rootFeature, maxDepth, bType, varargin{:}); % overwrite obj.tree created above
		end
	
	end

	methods (Access='protected')

		function [cn obj] = next(obj, pn)
		
			% if pn has dummy or no children yet, we know it has more children so we must add one randomly to the bandit at pn (we're sure it will be selected later because its UCB value will be +infinity)
			if (obj.tree.hasDummyChild(pn) || obj.tree.firstChild(pn)==0)
				obj.newChild(pn);
			end
			
			children = obj.tree.getChildren(pn);
			[discard i] = max(obj.tree.U(children));
			cn = children(i);
		
		end

	end
	
end