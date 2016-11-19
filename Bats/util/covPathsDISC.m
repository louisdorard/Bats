function [A, B] = covPathsDISC(logtheta, x, z)

% x and z are matrices of paths given as h. vectors
% a dummy path will be finished by a sequence of NaN values, so that it has the same length as a normal path
%
% For more help on design of covariance functions, try "help covFunctions".

function k = h2k(h)
% gives the kernel value for paths that have h nodes in common
	gamma = exp(logtheta);
	k = (1 - gamma.^(2.*h)) ./ (1 - gamma.^2);
end

function dk = h2dk(h)
% gives the derivative of the kernel function above with respect to its hyperparameter gamma
	gamma = exp(logtheta);
	dk = 2./gamma .* (((h-1).*gamma.^2-h) .* gamma.^(2.*h) + gamma.^2) / (1 - gamma.^2).^2;
end

if (nargin==0)
	A = '1'; % report number of parameters
	return;
end

if (nargin==2 || (nargin==3 && nargout==1)) % compute covariance
	A = covPathsLIN([], x);
	if (nargin==2)
		A = h2k(A);
	else
		% z is not important: there is only one element in logtheta
		A = h2dk(A);
	end
else
	% 3 arguments in and 2 out
	% -> compute test set covariances
	[A, B] = covPathsLIN([], x, z)
	A = h2k(A);
	B = h2k(B);
end

end