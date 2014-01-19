% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function
clear;

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.00001;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

DEBUG = false;
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

if DEBUG == true
%     ingerval = zeros(2*length(ei.layer_sizes)+1);
    paraSize = [];
    for i = 1:length(ei.layer_sizes)
        if i == 1
            paraSize = [paraSize  ei.input_dim*ei.layer_sizes(1) ei.layer_sizes(1)];
        else
            paraSize = [paraSize  ei.layer_sizes(i-1)*ei.layer_sizes(i) ei.layer_sizes(i)];
        end
    end
    interval = zeros(length(paraSize)+1,1);
    interval(1) = 1;
    for i = 1:length(paraSize)
        interval(i+1) = sum(paraSize(1:i))+1;
    end
    interval(end) = sum(paraSize);
    
    average_error = grad_check(@supervised_dnn_cost, params, interval, 1000, ei, data_train, labels_train);
    assert(average_error < 0.000001, 'gradient is wrong');
end
%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
