function LC = offspringSum(seed, B, LA)
%OFFSPRINGSUM
%   Gives features of child nodes available after the last element of a given path.
%   Used by the tree search algorithm during the exploration of a tree. The nodes it outputs can be stored in a tree structure.
%
%   INPUTS
%      seed - this is used to initialise the seed of the random generator, so that we generate children "randomly", but always generate the same children when starting from the same node
%             different values of seed yield different trees
%      B    - branching factor
%      LA   - feature representations of nodes of a given path, so that we know where we come from (size dim x depth-of-last-node)
%            (note that we can't give a sequence of node indices since these are specific to the tree structure)
%
%   OUTPUTS
%      LC   - list of children feature representations
%
%   See also REWARDSUM, ENVIRONMENTTS.
	
	
	% 3 features.
	% The first 2 are the coordinates: depth of node d, and index between 1 and B^d.
	% The third is the intermediate reward value that should be given when going through that node.
	%
	% rootFeature should be [0;1;0]
	
	
	d = LA(1,end);
	id = LA(2,end);
	
	% we use randn but need to make sure that the procedure is deterministic
	% => we need to initialise the random generator seed based on the coordinates of the parent node
	
	if (d==0)
		seed = seed + 1;
	else
		seed = seed + B.^(d-1) + id;
	end
	seed = mod(seed, 2^32); % seed shouldn't be bigger than 2^32
	s = RandStream('mcg16807', 'Seed', seed);
	prevstream = RandStream.getDefaultStream();
	RandStream.setDefaultStream(s);
	
	LC = zeros(3,B);
	for i=1:B
		LC(1,i) = d + 1; % depth of parent node plus one
		LC(2,i) = (id - 1) .* B + i; % index of parent minus one, times B, plus i
		LC(3,i) = randn();
	end
	
	RandStream.setDefaultStream(prevstream);
	
end