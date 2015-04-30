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

function R = run_bcs( trnFtMat, trnLblMat, valFtMat, tstFtMat, comp, logchi, logsigma, maxiter, verbose )
%RUN_BCS Summary of this function goes here
%   Detailed explanation goes here

    if ~exist('comp','var'), comp = 1.0; end
    if ~exist('logchi','var'), logchi = -1; end
    if ~exist('logsigma','var'), logsigma = -2; end
    if ~exist('maxiter','var'), maxiter = 500; end
    if ~exist('verbose','var'), verbose = 0; end

    R = struct(); cnt=1;
    for i=1:numel(comp),
        num_labels = size(trnLblMat,2);
        dim_z = ceil(num_labels*comp(i));
        if comp(i)==1,
            phi = eye(dim_z, num_labels);
        else
            phi = 2*rand(dim_z, num_labels)-1;
        end
        phi = normc(phi); % col-wise normalization
        
        for j=1:numel(logchi),
            for k=1:numel(logsigma),
                chi   = 10^logchi(j);
                sigma = 10^logsigma(k);
                
                if verbose>0,
                    fprintf('BCS comp_level=%.2f, chi=%.2f, sigma=%.2f',comp(i),chi,sigma);
                end
                tic;

                model = train_bcs(trnFtMat,trnLblMat,phi,sigma,chi,maxiter);
                [my.val,mz.val] = test_bcs(valFtMat,model,maxiter);
                [my.tst,mz.tst] = test_bcs(tstFtMat,model,maxiter);

                R(cnt).name = 'BCS';
                R(cnt).model = model;
                R(cnt).params.chi = chi;
                R(cnt).params.sigma = sigma;
                R(cnt).params.comp_level = comp(i);                 
                R(cnt).pred.my.val = my.val; 
                R(cnt).pred.my.tst = my.tst; 
                cnt = cnt + 1;
                
                time = toc;
                if verbose>0,
                    fprintf(', time=%.2f\n', time);
                end
            end
        end
    end         
end

