function [out1, out2] = gpr2(logtheta, covfunc, x, y, xstar, fix)
%GPR2
%   Same as gpr but for a covfunc which is the sum of a covSEard or covSEiso and a covNoise, in which the hyperparameters indicated by "fix" are fixed.
%   As a result, we set the derivatives with respect to these parameters to 0 at the end of this function.
%
%   See also GPML/GPR.

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n, D] = size(x);
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
  error('Error: Number of parameters do not agree with covariance function')
else
	2+2;
end

K = feval(covfunc{:}, logtheta, x);    % compute training set covariance matrix

L = chol(K)';                        % cholesky factorization of the covariance
alpha = solve_chol(L',y);

if (nargin<=4 || isempty(xstar)) % if no test cases, compute the negative log marginal likelihood

  out1 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);

  if nargout == 2               % ... and if requested, its partial derivatives
    out2 = zeros(size(logtheta));       % set the size of the derivative vector
    W = L'\(L\eye(n))-alpha*alpha';                % precompute for convenience
    for i = 1:length(out2)
      out2(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2;
    end
  end

elseif (~isempty(xstar))                    % ... otherwise compute (marginal) test predictions ...

  [Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     %  test covariances

  out1 = Kstar' * alpha;                                      % predicted means

  if nargout == 2
    v = L\Kstar;
    out2 = Kss - sum(v.*v)'; % sum(v.*v) is equivalent to v'*v
  end  

end

if (nargin==6)
	% set derivatives to 0 for the parameters which are fixed
	out2(fix) = 0;
end