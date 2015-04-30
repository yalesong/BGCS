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

function [ trn,val,tst ] = split_data( data, nfolds, no_val )
%SPLIT_DATA Splits data into train/valid/test splits. Subject independent.
%   Detailed explanation goes here
    
    if ~exist('no_val','var'), no_val = false; end

    subjects = unique({data.subject});
    nsubjects = numel(subjects);
    fold_size = round(nsubjects/nfolds);

    trn = cell(1,nfolds);
    val = cell(1,nfolds);
    tst = cell(1,nfolds);
    
    for fold=1:nfolds,
        sind_tst = fold_size*(fold-1)+1 : min(nsubjects,fold_size*fold);
        if no_val
            sind_val = sind_tst;
        else
            if mod(fold,nfolds)==0
                sind_val = 1 : fold_size;
            else
                sind_val = fold_size*fold+1 : min(nsubjects,fold_size*(fold+1));
            end
        end
        sind_trn = setdiff(1:nsubjects,[sind_val sind_tst]);
        
        ind_trn = ismember({data.subject},subjects(sind_trn));
        ind_val = ismember({data.subject},subjects(sind_val));
        ind_tst = ismember({data.subject},subjects(sind_tst));
        
        trn{fold} = data(ind_trn);
        val{fold} = data(ind_val);
        tst{fold} = data(ind_tst);
    end
end

