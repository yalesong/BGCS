%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bayesian Group-Sparse Compressed Sensing Toolbox  V1.0.0
%
%   Permissions granted under the MIT License (MIT)
%   Copyright (c) 2015 Yale Song (yalesong@csail.mit.edu) 
%
% Please cite the following paper if you end up using the code:
%
%   Yale Song, Daniel McDuff, Deepak Vasisht, and Ashish Kapoor.
%   "Exploiting Sparsity and Co-occurrence Structure for Action Unit
%   Recognition," IEEE FG 2015.
%   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change log
%
% Apr 30, 2015: Initial release
%

function model=train_bcs(train_features,train_labels,phi,sigma,chi,maxiter)
%train_features: N X D matrix where D is the number of features and N is the number of training points.
%train_labels: N X L matrix where L is the number of labels
%sigma,chi are model parameters
%maxiter is the number of iterations the max number of iterations
    [num_train, num_labels]=size(train_labels);
    num_features = size(train_features,2);
    dim_z = size(phi,1);
    X = train_features;
    Y = train_labels;

    sigmaw=(1/(sigma*sigma)*(X'*X)+sparse(eye(num_features)))\sparse(eye(num_features));

    mw = zeros(num_features,dim_z);
    mz = zeros(num_train,dim_z);
    
    num_iter = 0;
    not_converged = 1;
    param_old = [mw;mz];
    sigmaz = (1/(1/sigma/sigma+1/chi/chi))*eye(dim_z);
    while not_converged
        mw = 1/(sigma*sigma)*sigmaw*X'*mz;
        mz = (sigmaz*(1/(sigma*sigma)*mw'*X'+1/(chi*chi)*phi*Y'))';
        param_new = [mw;mz];
        err = max(max(abs(param_new-param_old)));
        
        if err<1e-2
            not_converged=0;
        else
            param_old=param_new;
        end
        num_iter=num_iter+1;
        if num_iter>maxiter
            not_converged=0;
        end
    end
    model.mw = mw;
    model.sigmaw = sigmaw;
    model.sigmaz = sigmaz;
    model.phi = phi;
    model.sigma = sigma;
    model.chi = chi;
end