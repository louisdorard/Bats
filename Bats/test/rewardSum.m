function r = rewardSum(gamma, signoise, LA)
%REWARDSUM
%   Gives the reward for a path, based on the node features vectors.
%   This is computed as the discounted sum of LA's components on its third row.
%   
%   INPUTS
%      gamma    - discount parameter
%      signoise - level of added noise
%      LA       - 1x1 cell containing a list of node feature vectors for this path, excluding the root (size 3 x maxDepth)
%
%   OUTPUTS
%      r        - reward
%
%   See also OFFSPRINGSUM, ENVIRONMENTTS.

LA = LA{1};

r = 0;
for i=1:size(LA,2)
	r = r + gamma.^(i-1).*LA(3,i);
end

r = r + signoise.*randn();

end