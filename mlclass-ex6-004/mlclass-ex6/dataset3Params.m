function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_mean= double(size(yval,1));

list= [0.01 0.03 0.1 0.3 1 3 10 30];

for i_C= list
	for i_sigma= list
		model= svmTrain(X, y, i_C, @(x1, x2) gaussianKernel(x1, x2, i_sigma));
		% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions= svmPredict(model, Xval);
		now_mean= mean(double(predictions ~= yval));

		i_C 
		i_sigma
		now_mean
		min_mean

		if(now_mean < min_mean)
			C= i_C;
			sigma= i_sigma;
			min_mean= now_mean;
		end
	end
end



% =========================================================================

end
