function Kc = normalise(K)

%function Kc = normalise(K)
%
% Normalises the kernel K
%
%INPUTS
% K = the non-normalised kernel K
%
%OUTPUTS
% Kc = the normalised kernel
%
%
%For more info, see www.kernel-methods.net


% original kernel matrix stored in variable K
% output uses the same variable K
% D is a diagonal matrix storing the inverse of the norms
D = diag(1./sqrt(diag(K)));
Kc = D * K * D;