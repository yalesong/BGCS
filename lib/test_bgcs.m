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

function [my, mz]=test_bgcs(test_features,model,max_iter,G)
%Test the model on testset.
%sigmaz,K,mw: outputs of train_data
%sigma,chi: same as the ones used in train_data
%max_iter: Maximum number of iterations you want to go to

    sigmaz   = model.sigmaz;
    mw       = model.mw;
    phi      = model.phi;
    sigma    = model.sigma;
    chi      = model.chi;
    num_test = size(test_features,1);
    [dim_z,num_labels] = size(phi);

    num_groups = size(G,1);
    ic_g = cell(1,num_groups); % index cell group
    for g=1:num_groups,
        ic_g{g} = find(G(g,:)>0);
    end
    iv_g = cell2mat(ic_g); 
    iic_g = mat2cell(1:numel(iv_g),1,cellfun(@(x) numel(x), ic_g));
    
    phi_g = zeros(size(phi,1),numel(iv_g));
    for i=1:numel(iv_g),
        phi_g(:,i) = phi(:,iv_g(i));
    end
    
    X = test_features;
    a_init = 1e-6*ones(num_test,num_groups);
    b_init = 1e-6*ones(num_test,num_groups);
    a = a_init;
    b = b_init;

    mz = X*mw;
    my = zeros(num_test,num_labels);
    my_g = zeros(num_test,numel(iv_g));

    num_iter = 0;
    not_converged = 1;
    param_old=[my mz];
    while not_converged
        for i=1:num_test
            mz(i,:)=(sigmaz*(1/sigma/sigma*mw'*X(i,:)'+1/chi/chi*phi_g*my_g(i,:)'))';
            for g=1:num_groups
                E_alpha(iic_g{g}) = min(1e+10,max(1e-10,a(i,g)/b(i,g)));
            end
            sigmay_g = (diag(E_alpha)+1/chi/chi*(phi_g'*phi_g))\eye(numel(iv_g));
            my_g(i,:)=(1/chi/chi*sigmay_g*phi_g'*mz(i,:)')'; 
            a(i,:) = a_init(i,:);
            b(i,:) = b_init(i,:);
            for g=1:num_groups,
                a(i,g)=a(i,g)+0.5*numel(ic_g{g});
                b(i,:)=b(i,:)+0.5*(my_g(i,ic_g{g})*my_g(i,ic_g{g})'+trace(sigmay_g(ic_g{g},ic_g{g})));
            end
        end
        
        %  mw=sigmaw*X'*mz;
        my = zeros(num_test,num_labels);
        for i=1:numel(iv_g),
            my(:,iv_g(i)) = my(:,iv_g(i)) + my_g(:,i);
        end

        param_new=[my mz];
        num_iter=num_iter+1;
        if(num_iter>=max_iter)
            not_converged=0;
        end
        err=max(max(abs(param_old-param_new)));
        %fprintf('iter=%d, error = %f\n', num_iter,err);
        if(err<1e-2)
            not_converged=0;
        else
            param_old=param_new;
        end
    end
end