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
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
% This gives a probability of y = +1
prob = exp(lp);

disp('Saving to prob.mat in current folder:')
disp(pwd)

save('prob.mat', 'prob');
save('b.mat', 'b');
