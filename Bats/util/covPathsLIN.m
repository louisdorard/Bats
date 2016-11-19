function [A, B] = covPathsLIN(logtheta, x, z);

% x and z are matrices of paths given as h. vectors
% a dummy path will be finished by a sequence of NaN values, so that it has the same length as a normal path
%
% For more help on design of covariance functions, try "help covFunctions".

function n = nCommonNodes(a, b, l)
% gives the number of nodes in common between a and b of length l
	s = sum(a==b);
	n1 = sum(isnan(a));
	n2 = sum(isnan(b));
	if (n1==n2 && s+n1==l) % this means that x(i,:) and x(j,:) are the same
		n = l; % if we would do an intersect, we would discard the NaNs that might be present in dummy nodes' paths, but the kernel product of a dummy with itself should be the length of the path
	else
		n = s;
	end
end

if nargin == 0, A = '0'; return; end % report number of parameters

if nargin == 2 % compute covariance
	n = size(x,1);
	l = size(x,2);
	A = zeros(n); % covariance matrix for the data points in x
	for i=1:n
		for j=i:n
			A(i,j) = nCommonNodes(x(i,:), x(j,:), l);
			A(j,i) = A(i,j);
		end
	end
elseif nargout == 2 % compute test set covariances
	if (size(x,2)~=size(z,2))
		error('Dimension mismatch');
	end
	n = size(x,1);
	l = size(x,2); % this is also be equal to size(z,2)
	nn = size(z,1);
	A = zeros(nn,1); % self covariances
	B = zeros(n,nn); % cross covariances
	for j=1:nn
		A(j,1) = l; % self covariances all have the same value
	end
	for i=1:n
		for j=1:nn
			B(i,j) = nCommonNodes(x(i,:), z(j,:), l);
		end
	end
else
	A = 0; % derivative matrix
end

end