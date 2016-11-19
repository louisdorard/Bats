function Cti = covinv(K, signoise, Ctiprev, k1)
%COVINV
%   Computes the inverse of the covariance matrix with Gaussian additive noise.
%   This is done either from scratch, or based on the previous covariance matrix for all training points except the last one.
%
%   INPUTS
%      K        - if Ctiprev and k1 are given, this represents the vector of kernel products between the last training point and the others
%                 otherwise, this represents the kernel/covariance matrix
%      signoise - noise standard deviation in the Gaussian additive noise model
%      Ctiprev  - (optional) previous covariance matrix inverse
%      k1       - (optional) value of kernel product between last point and itself
%
%   OUTPUTS
%      Cti      - inverse covariance matrix


if (size(signoise,2)>1)
    signoise = signoise(1);
end

t=size(K,1);

if (nargin<4)

	% if Ctiprev not given, K represents the kernel matrix
	Cti = inv(K + diag(signoise)^2.*eye(t));
	
else

    % if Ctiprev is given, the inverse of the covariance matrix can be obtained
	% with the block inversion lemma
	
	k = K;
	s2 = k1 + signoise.^2 - k' * Ctiprev * k;
	
	A = Ctiprev + 1./s2 .* Ctiprev * k * (Ctiprev * k)';
	d = -1./s2 .* Ctiprev * k;
	e = 1./s2;
	
	Cti = [A d; d' e];

end