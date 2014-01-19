function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
avg = mean(x, 1);    
x = x - repmat(avg, size(x, 1), 1);
sigma = x * x' / size(x, 2);
[u,s,v] = svd(sigma);
% xRot = u' * x;

xPCAWhite = diag(1./sqrt(diag(s)+epsilon)) * u' * x;
Z = u * xPCAWhite;
end