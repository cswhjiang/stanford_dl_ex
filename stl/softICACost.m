%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
Y = W*x;
m = size(x,2);
cost = params.lambda * sum(sum(sqrt(Y.^2 + params.epsilon))) + 0.5*norm(W'*Y - x,'fro')^2/m;

Wgrad = params.lambda *(Y./sqrt(Y.^2+params.epsilon))*x' + (W*W'*Y*x' + Y*Y'*W - 2*Y*x')/m;


% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
end