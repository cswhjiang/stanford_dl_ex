function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
poolLen = floor(convolvedDim / poolDim);
rb = 0;
re = 0;
cb = 0;
ce = 0;

for i = 1 : numFilters
    for j = 1 : numImages
        for r = 1 : poolLen
            for c = 1 : poolLen
                rb = 1 + poolDim * (r-1);
                re = poolDim * r;
                cb = 1 + poolDim * (c-1);
                ce = poolDim * c;
%                 blockFeatures = convolvedFeatures(i, j, rb : re, cb : ce);
                pooledFeatures(r, c, i, j) = mean(mean(convolvedFeatures( rb : re, cb : ce,i,j)));
            end
        end
    end
end
end

