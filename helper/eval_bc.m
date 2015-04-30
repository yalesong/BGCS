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

function [ r ] = eval_bc( ytrue, ypred, target )        
%EVAL_BC Computes various performance metrics for binary classification.
% Input
%     ytrue : ground truth labels
%     ypred : predicted labels
%     target: target class (default 1)
% Output
%     r     : contains performance metrics

    if ~exist('target','var'),
        target = 1;
    end
    class_order = [target setdiff([0 1],target)];
    
    % Compute confusion matrix
    % cm = tp (1,1), fn (1,2), fp (2,1), tn (2,2)
    cm = confusionmat(ytrue,ypred,'order',class_order); 
    r.confmat = cm;
    
    % Compute precision and recall
    r.prec = cm(1,1)/(cm(1,1)+cm(2,1)); % tp/(tp+fp)
    r.rec  = cm(1,1)/(cm(1,1)+cm(1,2)); % tp/(tp+fn)            
    
    % true negative rate: tn / (tn+fp)
    r.tnn = cm(1,1)/(cm(1,1)+cm(2,1)); 
    
    % accuracy: (tp+tn)/(tp+tn+fp+fn)
    r.acc = (cm(1,1)+cm(2,2))/sum(cm(:)); 
    
    % f1 score: 2*(prec*rec)/(prec+rec)
    r.f1  = 2*(r.prec*r.rec)/(r.prec+r.rec);     
    
    if isnan(r.prec), r.prec=0; end
    if isnan(r.rec), r.rec=0; end
    if isnan(r.tnn), r.tnn=0; end
    if isnan(r.acc), r.acc=0; end
    if isnan(r.f1), r.f1=0; end
end

