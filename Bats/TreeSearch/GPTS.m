classdef GPTS < BanditTS
%GPTS - Gaussian Processes Tree Search algorithm
%   This is a single-bandit tree search algorithm: it implements the superclass methods with a normal tree and only one bandit algorithm (GPB).
%   
%   In the GPB instance, arms are dummy or leaf nodes, represented by the paths that lead to them.
%   Arms' labels are nodes' indices. They are used for identification.
%   Arms' feature vectors are v. vectors of node indices (up to the depth of the dummy/leaf node) and NaN entries (up to the maximum depth),
%   i.e. they are a concatenation of a transposed path and NaN entries. They are used for computing kernel products.
%   Note that GPB doesn't consider the node feature representations, but these are used by the superclass when calling offspring.
%
%   GPTS Properties
%      b           - GPB instance
%      learnHyper  - specifies how often the hyperparameters should be learnt
%                    (0 when we don't want to learn them)
%
%   See also BANDITTS.

	properties (GetAccess='public', SetAccess='protected')
		b; % GPB bandit algorithm
		learnHyper;
    end

	methods (Access='public')
	
		function obj = GPTS(rootFeature, offspring, reward, maxDepth, rbest, ker, signoise, S, learnHyper)
            obj = obj@BanditTS(rootFeature, offspring, reward, maxDepth, rbest, 'depth-first');
			L = obj.randomWalk(1); % leaf and dummy nodes created during random walk
			obj.b = GPB(ker, signoise, L, obj.tree.getPath(L)', 0.05, 'online1'); % feature vectors are the paths to the nodes created by the random walk; default value of delta=0.05 is used
			if (nargin>=8)
				obj.b.S = S;
				if (nargin>=9)
					obj.learnHyper = learnHyper;
				else
					obj.learnHyper = 0;
				end
			end
			% IDEA try b.chooseNew = true;
        end
	
	end
	
	methods (Access='protected')
        
		function obj = train(obj, path, y)
			
			cn = GPTS.lastPathNode(path);
			obj.b.train(cn, y, 'label'); % the arm to put in training is identified by the index of the corresponding node
			if (obj.learnHyper)
				% learn hyperparameters every learnHyper data points
				if (mod(obj.b.ntr, obj.learnHyper)==0)
					obj.b.learnHyper();
				end
			end
		end
		
		function p = best(obj, np)
			% chooses the np best first moves (np should be smaller than maxDepth!)
			% here, best is as in, best M value (we could also take best U or best lower confidence bound, which are known since we are using ucb-type algorithms here)
			
			[A i] = max(obj.b.M); % i is the index of the arm that has highest M value
			p = obj.explorePathFromBandit(i); % this gives a path to a leaf (no NaNs)
			p = p(1:np);
			
		end

		function [path obj] = search(obj)
		%SEARCH
		%
		%   OUTPUTS
		%      path - h. vector of node indices
			
			path = obj.explorePathFromBandit(obj.b.choose());

		end
		
		function path = explorePathFromBandit(obj, i)
			% path doesn't include the root node
			
			path = obj.b.features(:,i)';
			cn = GPTS.lastPathNode(path);
			if (obj.tree.isdummy(cn))
				L = obj.randomWalk(cn);
				if (~isempty(L))
					% add nodes created during the random walk, to the bandit problem
					for n=L % can be a dummy or a leaf
						obj.b.addArm(n, obj.tree.getPath(n)'); % label is the node index; the path to n is the feature representation of the corresponding arm
					end
					% the dummy doesn't exist anymore (transformed into regular node) and it is the path to the new leaf node that we should output
					cn = L(end);
					path = [obj.tree.getPathTo(cn) cn];
				end
			end
			
		end
		
		function [L obj] = randomWalk(obj, cn)
		%RANDOMWALK
		%   Do a random walk from node cn to a leaf. cn can be a dummy or the root.
		%   - If it is a dummy, cn is transformed into a regular node.
		%   - If it is the root, then the tree should be empty (except for the root).
		%   The random walk creates new nodes and dummies as it goes down the tree.
		%
		%   INPUTS
		%      cn       - node where to start the random walk from
		%                 cn's ancestors (if any) have been explored
		%      maxDepth - depth at which we must stop the walk
		%
		%   OUTPUTS
		%      L        - list of leaf and dummy nodes created during the random walk
		%                 there can be only one leaf node, which is given by L(end)
		%      path     - path to the leaf node created at the end of the random walk

			
			L = [];
			
			%%%%
			% first, get the depth of cn
			%
		    % if the input is a dummy, cn becomes a new sibling of this dummy
		    % otherwise, cn is unchanged
		    %%%%

			if (obj.tree.parent(cn)==0)
				cnd = 0;
				if (~isempty(obj.tree.getChildren(cn)))
					error('If cn is the root, it shouldnt have any children');
				end
			elseif (obj.tree.isdummy(cn))
				cnd = length(obj.tree.getPathTo(cn))+1;
		        [cn2 discard] = obj.newChild(obj.tree.parent(cn)); % obj.tree.parent(cn) having a dummy as cn, we know we're in case 1 or 4 of createNode
				% being in case 1 or 4, we know that no new dummy node has been created, hence dn=0 and we don't need to add it to L
				% cn might change but not its depth
				if (cn2~=cn && cnd==obj.tree.maxDepth) % a new leaf node has been created!
					L = [L cn2];
				else
					% this means that cn was a dummy now turned into a regular node, so it's not a playable arm anymore (note: it hasn't been played at all)
					% but it stays in the bandit's list of arms -> we must mark it as not playable
					obj.b.playable = setdiff(obj.b.playable, find(obj.b.labels==cn2));
				end
				cn = cn2;
			else
				error('cn should be a dummy or the root of the tree');
			end


			%%%%
			% finally, random walk FROM cn, stopping at maxDepth
			%%%%
			
			% if cnd is equal to maxDepth, then cn was a dummy that was transformed into a leaf, so no node has been created and L should stay empty
			if (cnd<obj.tree.maxDepth)
				while (cnd<obj.tree.maxDepth) % we can't call createNode when at depth maxDepth
					[cn dn] = obj.newChild(cn); % we are necessarily in case 3 of createNode (if cn is root, this is clear; otherwise, cn has just been transformed into a regular node which will have at least 2 children, if we assume that the branching factor of the tree is always 2 or bigger)
					L = [L dn]; % in case 3, we know that a dummy node has been created (hence dn~=0) 
					cnd = cnd+1;
				end
				L = [L cn]; % cnd is now equal to maxDepth and cn is therefore a leaf node -> it should be added to L
			end

		end
		
	end
	
	methods (Static)
		
		function cn = lastPathNode(path)
			path = intersect(path, path); % ditch the NaNs, if any
			cn = path(end); % get the last node's index
		end
		
	end
	
end