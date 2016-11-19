function K = gaussianGramMatrix(X, sigma)
%GAUSSIANGRAMMATRIX
%   Computes the Gram matrix for the given training set, for the Gaussian kernel
%
%   INPUTS
%      X     - set of training points
%      sigma - width of the Gaussian kernel
%
%   OUTPUTS
%      K     - Gram Matrix


	n = size(X, 2);
	K = zeros(n);
	for i=1:n
		for j=i:n
			K(i,j) = exp (-norm(X(:,i)-X(:,j),2).^2./(2.*sigma.^2) );
			K(j,i) = K(i,j);
		end
	end
	
	
end