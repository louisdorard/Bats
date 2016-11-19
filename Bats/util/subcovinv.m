function Cti = subcovinv(Ctiall)
%SUBCOVINV
%   Computes the inverse of the covariance matrix when removing first datapoint.
%
%   INPUTS
%      Ctiall   - inverse covariance matrix for all data points
%
%   OUTPUTS
%      Cti      - inverse covariance matrix for all data points but the first one


t = size(Ctiall,1);

if (nargin<3)

	% using the block inversion lemma
	
	e = Ctiall(1,1);
	d = Ctiall(2:t,1);
	A = Ctiall(2:t,2:t);
	
	Cti = A-1./e.*d*d';

end