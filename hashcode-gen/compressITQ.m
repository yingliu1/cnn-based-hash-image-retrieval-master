function varargout = compressITQ(bit, X, varargin)

% code for converting data X to binary code C using ITQ
% Input:
%       X: n*d data matrix, n is number of images, d is dimension
%       bit: number of bits
% Output:      
%       C: n*bit binary code matrix

% center the data, VERY IMPORTANT for ITQ to work
sampleMean = mean(X,1);
X = X - repmat(sampleMean,size(X,1),1);

% PCA
[pc, ~] = eigs(cov(X), bit);
XX = X * pc;

% ITQ to find optimal rotation
% default is 50 iterations
% C is the output code
% R is the rotation found by ITQ
[C, R] = ITQ(XX,50);
varargout{1} = C;

%% detect if there is test data
if nargin >2
    Y = varargin{1};
    Y = Y - repmat(sampleMean,size(Y,1),1);
    Y = Y * pc;
    Z = Y * R;      
    UX = ones(size(Z,1),size(Z,2)).*-1;
    UX(Z>=0) = 1;
    B = UX;
    B(B<0) = 0;
    varargout{2} = B;
end

