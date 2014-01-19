function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%

for i = 1:length(hAct)
    if i == 1
        hAct{1} = sigmoid( bsxfun(@plus, stack{1}.W * data , stack{1}.b));
    elseif i == length(hAct)
        Aout = exp(bsxfun(@plus, stack{i}.W * hAct{i-1}, stack{i}.b));
        hAct{end} = bsxfun(@rdivide, Aout, sum(Aout));
        clear Aout;
    else
        hAct{i} = sigmoid(bsxfun(@plus, stack{i}.W * hAct{i-1}, stack{i}.b));
    end
end
pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

% the last layer is logistic activation, all previous layers is sigmoid
% activation
%  TODO: support other activations

m = size(data,2);
groundTruth = full(sparse(labels, 1:m, 1));


% for i = 1:length(hAct)
%     if i == 1
%         hAct{1} = sigmoid( bsxfun(@plus, stack{1}.W * data , stack{1}.b));
%     elseif i == length(hAct)
%         Aout = exp(bsxfun(@plus, stack{i}.W * hAct{i-1}, stack{i}.b));
%         hAct{end} = bsxfun(@rdivide, Aout, sum(Aout));
%         clear Aout;
%     else
%         hAct{i} = sigmoid(bsxfun(@plus, stack{i}.W * hAct{i-1}, stack{i}.b));
%     end
% end


costObj = -sum(sum(groundTruth.*log(hAct{end})))/m;
costPenalty = 0;
for i = 1: length(ei.layer_sizes)
    costPenalty = costPenalty  + norm(stack{i}.W,'fro')^2;
end
cost  = costObj + (ei.lambda/2)*costPenalty;


%%

Delta = cell(numHidden+1, 1);
for i = length(Delta):-1:1
    if i == length(Delta)
       Delta{i} =  -(groundTruth - hAct{i}); 
    else
       Delta{i} = (stack{i+1}.W'*Delta{i+1}).*(hAct{i}.*(1-hAct{i})); % d_h*m
    end
    
end


for i = 1:length(gradStack)
    if i == 1
        gradStack{1}.W = Delta{1}*data'/m + ei.lambda * stack{1}.W;
        gradStack{1}.b = sum(Delta{1},2)/m;
    else
        gradStack{i}.W = Delta{i}*hAct{i-1}'/m + ei.lambda * stack{i}.W;
        gradStack{i}.b = sum(Delta{i},2)/m;
    end
end


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



