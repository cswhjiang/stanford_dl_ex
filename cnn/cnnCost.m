function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses, filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images
lambda = 3e-3; % weight decay parameter    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
convolvedFeatures = cnnConvolve(filterDim, numFilters, images, Wc, bc); 
activationsPooled = cnnPool(poolDim, convolvedFeatures);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
%Wd=(numClasses,hiddenSize)
M = Wd*activationsPooled+repmat(bd,[1,numImages]); 
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
probs = bsxfun(@rdivide, M, sum(M)); %why rdivide?
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
groundTruth = full(sparse(labels, 1:numImages, 1));
cost = -1./numImages*groundTruth(:)'*log(probs(:))+(lambda/2.)*(sum(Wd(:).^2)+sum(Wc(:).^2)); %

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

% 网络结构: images--> convolvedFeatures--> activationsPooled--> probs
% Wd = (numClasses,hiddenSize)
% bd = (hiddenSize,1)
% Wc = (filterDim,filterDim,numFilters)
% bc = (numFilters,1)
% activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
% convolvedFeatures = (convDim,convDim,numFilters,numImages)
% images(imageDim,imageDim,numImges)
delta_d = -(groundTruth-probs); % softmax layer's preactivation,每一个样本都对应有自己每层的误差敏感性。
Wd_grad = (1./numImages)*delta_d*activationsPooled'+lambda*Wd;
bd_grad = (1./numImages)*sum(delta_d,2); %注意这里是要求和
delta_s = Wd'*delta_d; %the pooling/sample layer's preactivation
delta_s = reshape(delta_s,outputDim,outputDim,numFilters,numImages);

%进行unsampling操作，由于kron函数只能对二维矩阵操作，所以还得分开弄
%delta_c = convolvedFeatures.*(1-convolvedFeatures).*(1./poolDim^2)*kron(delta_s, ones(poolDim)); 
delta_c = zeros(convDim,convDim,numFilters,numImages);
for i=1:numImages
    for j=1:numFilters
        delta_c(:,:,j,i) = (1./poolDim^2)*kron(squeeze(delta_s(:,:,j,i)), ones(poolDim));
    end
end
delta_c = convolvedFeatures.*(1-convolvedFeatures).*delta_c;

% Wc_grad = convn(images,rot90(delta_c,2,'valid'))+ lambda*Wc;
for i=1:numFilters
    Wc_i = zeros(filterDim,filterDim);
    for j=1:numImages  
        Wc_i = Wc_i+conv2(squeeze(images(:,:,j)),rot90(squeeze(delta_c(:,:,i,j)),2),'valid');
    end
   % Wc_i = convn(images,rot180(squeeze(delta_c(:,:,i,:))),'valid');
    % add penalize
    Wc_grad(:,:,i) = (1./numImages)*Wc_i+lambda*Wc(:,:,i);
    
    bc_i = delta_c(:,:,i,:);
    bc_i = bc_i(:);
    bc_grad(i) = sum(bc_i)/numImages;
end

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
