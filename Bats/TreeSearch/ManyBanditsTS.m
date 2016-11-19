classdef ManyBanditsTS < BanditTS
%MANYBANDITSTS abstract class
%   To be used for many-bandits tree search algorithms: the superclass methods are implemented by calling the bandit algorithms that are stored at each node of the BTree structure.
%   The type of bandit algorithms to be used (UCB, GPB, etc.) is not specified here.
%
%   Searching consists of several iterations of going down the tree by using the next method to determine which child to go to next (through the 'next' method which is left unimplemented).
%
%   See also BANDITTS, BTREE.
    
    methods
        
        function obj = ManyBanditsTS(rootFeature, offspring, reward, maxDepth, rbest, growMethod)
            obj = obj@BanditTS(rootFeature, offspring, reward, maxDepth, rbest, growMethod);
        end
        
    end
	
	methods (Access='protected')
        
		function obj = train(obj, path, y)
			% the bandit algorithms of all nodes on the path have to be updated with this new training example
			% -> we back-propagate the obtained reward from the leaf to the root
			for i = length(path):-1:2
				cn = path(i);
				pn = path(i-1);
				obj.tree.trainBandit(pn, cn, y); % means that bandit at node pn needs to add following info to training: arm identified by label cn got a reward of y
			end
			obj.tree.trainBandit(1, path(1), y); % we also need to update the root's bandit algorithm, as well as B(path(1))!
		end
		
		function p = best(obj, np)
			% chooses the np best first moves (np should be smaller than maxDepth!)
			% here, best is as in, best M value (we could also take best U or best lower confidence bound, which are known since we are using ucb-type algorithms here)
			p = [];
			pn = 1;
			for d=1:np
				if (obj.tree.firstChild(pn) == 0) break; end % what if there's no child, hence no bandit here at cn? -> we stop here
				[discard i] = max(obj.tree.bandit{pn}.M);
				cn = obj.tree.bandit{pn}.labels(i);
				p = [p cn];
			end
			p = obj.tree.features(:,p);
		end
		
		% UCT does it by going down the tree iteratively, whereas GPB would choose a path in one go
		function [path obj] = search(obj)
			pn = 0; cn = 1; d = 0;
			path = []; % don't include the root in the path
			stop = false;
			while ~(stop)
                pn = cn;
				cn = obj.next(cn); d = d+1; % definition of current node and its depth; next(cn) deals with choosing node and creating new node if necessary (based on offspring)
				path = [path cn]; % we need to output a h. vector
				if (d>=obj.tree.maxDepth) % we always stop at maxDepth
					stop = true;
				end
                b = obj.tree.bandit{pn};
                % we haven't (re)trained b on cn yet (this is done at the end
                % of the search), so in the case of iterative-deepening we
                % have to stop if cn has never been seen before, i.e. if it
                % is not in the training set of b
				if (strcmp(obj.growMethod, 'iterative-deepening') && isempty(find(b.labels(b.Tr.X)==cn, 1))) % we stop even earlier if doing iterative-deepening
					stop = true;
				end
			end
		end

		% Chooses a child of pn to go to in our search of the tree. We choose an existing child, or we can also create a new child and choose it.
		[cn obj] = next(obj, pn)
	
	end

end