function [W] = DRAGD(Train_Ma,Z,Train_Lab,lambda1,lambda2,lambda3,miu,rho,max_iter,max_iter_diffusion)
% The code is written by Jie Wen, if you have any questions, please contact
% jiewen_pr@126.com
% Note: if you use the code, please cite:
% J Wen, S Deng, L Fei, et al., Discriminative Regression with Adaptive Graph Diffusion[J]. TNNLS, 2022.

nnClass = length(unique(Train_Lab));
label = unique(Train_Lab);
T  = bsxfun(@eq, Train_Lab, label');
T  = double(T');
A = Z;
S = Z;

C2 = 0;
XX = Train_Ma*Train_Ma';
Z_bar = diag(1./sqrt(sum(Z,1)))*Z*diag(1./sqrt(sum(Z,1)));
W = T*Train_Ma'/(XX+lambda3*eye(size(XX,1)));
for iter = 1:max_iter  
    W_old = W;
    A_old = A;
    S_old = S;
    % --------- A ---------- %
    New_X = W*Train_Ma;
    G = EuDist2(New_X',New_X',0);
    V = S-C2/miu-lambda1*0.5/miu*G;
    U = (V+V')*0.5;
    clear New_X V
    Nsap = size(U,1);
    ac1 = ones(Nsap,1);
    A = U+(Nsap+ac1'*U*ac1)/Nsap^2*ones(Nsap,Nsap)-(U*ones(Nsap,Nsap)+ones(Nsap,Nsap)*U)/Nsap;
    A = A-diag(diag(A));
    for ii = 1:10
        if min(A(:))<0            
            A = max(A,0);
            U = (A+A')*0.5;
            A = U+(Nsap+ac1'*U*ac1)/Nsap^2*ones(Nsap,Nsap)-(U*ones(Nsap,Nsap)+ones(Nsap,Nsap)*U)/Nsap;
            A = A-diag(diag(A));
        else
            break
        end
        
    end
    A = max(A,0);
    A = A-diag(diag(A));
    % --------- S ---------- %
    for ii = 1:max_iter_diffusion 
        H = A+C2/miu;
        alpha = lambda2/(lambda2+miu);
        S = alpha*(Z_bar*S*Z_bar') + (1-alpha)*H;   
    %     S = max(S,0);   
        S = S-diag(diag(S)); 
        if max(max(abs(W-W_old)))<1e-4
            break
        end
    end
    % ---------- T ----------- %    
    R = W*Train_Ma;
    T1 = zeros(nnClass,size(Train_Ma,2));
    for ind = 1:length(Train_Lab)
         T1(:,ind) = (optimize_R(R(:,ind)', Train_Lab(ind)))';
    end
    T = T1;
    clear T1
    % -------- W ------------- %
    A = (A+A')*0.5;
    La = diag(sum(A,1))-A;
    W = 2*T*Train_Ma'/(2*lambda1*Train_Ma*La*Train_Ma'+2*XX+lambda3*eye(size(Train_Ma,1)));    
    % ------- C1 C2 miu --------- %
    leq1 = A-S;
    C2 = C2+miu*leq1;
    miu = min(1e10,rho*miu);  
    % ---------- obj -------- %       
    L1 = max(max(abs(W-W_old)));
    L2 = max(max(abs(S-S_old)));
    L4 = max(max(abs(A-A_old)));
    L12 = max([L1,L2,L4]);
    obj(iter) = max(max(abs(leq1(:))),L12);
    if iter>3 && obj(iter)<1e-2
        iter
        break;
    end
    
end
end