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

function [my, mz]=test_bcs(test_features,model,max_iter)
%Test the model on testset.
%sigmaz,K,mw: outputs of train_data
%sigma,chi: same as the ones used in train_data
%max_iter: Maximum number of iterations you want to go to
    sigma  = model.sigma;
    sigmaz = model.sigmaz;
    mw     = model.mw;
    phi    = model.phi;
    chi    = model.chi;
    
    num_test = size(test_features,1);
    [dim_z, num_labels] = size(phi);
    X = test_features;
    
    a_init = 1e-6*ones(num_test,num_labels);
    b_init = 1e-6*ones(num_test,num_labels);
    a = a_init;
    b = b_init;
    
    mz = X*mw;
    my = zeros(num_test,num_labels);
    
    num_iter = 0;
    not_converged = 1;
    param_old = [my mz];
    while not_converged
        for i=1:1:num_test
            mz(i,:) = (sigmaz*(1/sigma/sigma*mw'*X(i,:)'+1/chi/chi*phi*my(i,:)'))';
            E_alpha = min(1e+10,max(1e-10,a(i,:)./b(i,:)));
            sigmay  = (diag(E_alpha)+1/chi/chi*(phi'*phi))\eye(num_labels);   
            my(i,:) = (1/chi/chi*sigmay*phi'*mz(i,:)')';
            a(i,:)  = a_init(i,:)+0.5;
            b(i,:)  = b_init(i,:)+0.5*(diag(sigmay)'+my(i,:).^2);
        end
      
        param_new = [my mz];
        num_iter = num_iter+1;
        if num_iter>=max_iter
            not_converged=0;
        end
        err = max(max(abs(param_old-param_new)));
        
        if(err<1e-2)
            not_converged=0;
        else
            param_old=param_new;
        end
    end
end

