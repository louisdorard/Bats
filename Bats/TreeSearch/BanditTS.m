classdef BanditTS < TreeSearchInterface
%BANDITTS
%   Defines the procedure shared by all bandit-based algorithms
%   to choose an optimal tree path.
%
%   BanditTS Properties:
%      e               - (Read-only) environment: gives rewards to paths, and 
%                        specifies the children of any given node
%      growMethod      - (Read-only) 'depth-first' or 'iterative-deepening'
%      t               - (Read-only) number of iterations that have been performed
%	                     = current iteration/time-step minus 1
%                        should match the t value of the environment
%      tree            - (Read-only) tree structure where the explored nodes are
%                        stored
%
%   BanditTS Methods:
%      BanditTS - constructor
%      choose           - choose a path after several bandit iterations
%
%   See also TREESEARCHINTERFACE, TREE.

    properties (GetAccess='public', SetAccess='protected')
		e;
		growMethod = 'depth-first';
		t = 0;
		tree;
    end
	
	
	methods (Access='public')
		
		function obj = BanditTS(rootFeature, offspring, reward, maxDepth, rbest, growMethod)
		%BANDITTS
		%   Constructor: initialises class properties, in particular the environment.
		%      rootFeature       - (Read-only) represents the state of the decision process
		%                        when the tree search is started
		%			e.g. for Go it can be the current Go board from which we want to know the optimal move
		%			e.g. for labelling, it is the (x_{i_0}, y_{i_0}) couple
			
			
			% offspring, reward = see EnvironmentTS comments; reward has to be able to give a reward for paths of any length if growMethod is iterative deepening
			% growMethod = iterative-deepening or depth-first
			
			obj.growMethod = growMethod;
			obj.e = EnvironmentTS(reward, offspring, rbest);
			obj.tree = Tree(rootFeature, maxDepth);
			
        end

		function [p d] = choose(obj, np, nit)
		%CHOOSE
		%   See superclass.
		%   We perform nit iterations. Each consists in searching the tree for a path to play, playing it and learning.
			
			if (nit>0)
				for i=1:nit
					path = obj.search(); % doesn't include the root node
					y = obj.e.play({obj.tree.features(:,path)}); % if we don't put this in a cell, it will look as several arms to be played
					obj.train(path, y);
					obj.t = obj.t + 1;
				end
			end
			p = obj.best(np);
			d = obj.e.R(end) ./ nit; % cumulative regret bound -> max regret bound
			
		end

    end
	
	
	methods (Abstract, Access='protected')
        
		% Do 1 search of the tree to find a path (h. vector of node indices) to play.
		% This implies the creation of nodes in the tree, as we are exploring new paths.
		% path shouldn't include the root node
		[path obj] = search(obj)
		
		% Train the learning algorithm when observing a sequence of node indices "path" and a reward "y"
		obj = train(obj, path, y)
		
		% Get the best np first moves
		% IDEA: best according to what? highest estimated reward, highest UCB, highest LCB?
		p = best(obj, np)
        
    end


	methods (Access='protected')
	
		function [cn dn] = newChild(obj, pn)
		%NEWCHILD
		%   Create a new child to pn in the tree structure,
		%   with feature randomly selected from the list of possible features given by offspring
		%   and that are not yet present in the tree structure.
		%
		%   Calling offspring requires to have access to the environment and to the tree to determine the list of ancestors of pn and their features.
		%   Determining the labels that are not yet present in the tree structure requires to determine the list of children of pn.
		%   Once its label has been determined, creating the new child in the tree is done through createNode.
		%   
		%   Used in UCT and GPTS
		%
		%   INPUTS
		%      pn  - parent node where we want to create the new child
		%
		%   OUTPUTS
		%      see Tree.createNode
		
			% we give the list of labels of ancestors, the children we already have, and we want a new child selected at random among the remaining children (and we want to know if it's gonna be the last one)
			LC1 = obj.tree.features(:,obj.tree.getChildren(pn));
			LA = obj.tree.features(:,[obj.tree.getPathTo(pn) pn]); % this is used when calling offspring to say where we are in the tree
			LC = obj.e.offspring(LA);
            % we need to take transposes because setdiff works on rows and not columns :(
            if (~isempty(LC1'))
                lc2 = setdiff(LC', LC1', 'rows')'; % assuming that each label is a vector, i.e. {[1;2;3]} for instance, we get a matrix of vectors corresponding to the unexplored children's labels
            else
                lc2 = LC;
            end
            % lc2 is a concatenation of (vertical) feature vectors
			nb = size(lc2,2);
			if (nb==1)
				isLastChild = 1;
			else
				isLastChild = 0;
			end
			% pick one column of lc2 at random and output it as l
			% we output vector, and not {vector}
			rd = rand();
			l = lc2(:,floor(1+rd.*nb));
			
			[cn dn] = obj.tree.createNode(pn, l, isLastChild);
		end
	
	end

end