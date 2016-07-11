% Include subdirectories to use GPML code
addpath(genpath('./'))

load('train.mat');
load('test.mat');

disp( size(x) )
disp( size(y) )
disp( size(t) )

% Train
meanfunc = @meanConst;
hyp.mean = 0;
covfunc = @covLIN;
hyp.cov = [];
likfunc = @likErf;

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);

[n, nn] = size(t)
% The main workhorse function call. Requres:
%   - hyp:    a hyperparameter struct  (depends on what mean, cov, lik we pick)
%   - inf:    the inference method
%   - mean:   the mean function        (prior)
%   - cov:    the covariance function  (prior)
%   - lik:    the likelihood function
%   - x:      train x
%   - y:      train y
%   - xs:     test x
%   - ys:     test y
%   => last output, 'lp', will contain the log-likelihood of each test label,
%      i.e., since we pass 'ones', the log-likelihood that that document is
%      relevant. We then only aggregate votes from relevant documents
%      (likelihood > 0.5).
%
% The input x consists of tf-idf representation of documents, and the label is
% a boolean indicating whether that document is relevant for a given vote. Yes,
% this means that for one document with 'k' votes, there are 'k' entries passed
% to Matlab, so there are duplicate input entries.
%
% From the documentation:
% [gp] does posterior inference, learns hyperparameters, computes the marginal
% likelihood and makes predictions. Generally, the gp function takes the
% following arguments: a hyperparameter struct, an inference method, a mean
% function, a covariance function, a likelihood function, training inputs,
% training targets, and possibly test cases. The exact computations done by the
% function is controlled by the number of input and output arguments in the
% call.
% In our case: 5 return args -> perform prediction
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
% This gives a probability of y = +1
prob = exp(lp);

disp('Saving to prob.mat in current folder:')
disp(pwd)

save('prob.mat', 'prob');

