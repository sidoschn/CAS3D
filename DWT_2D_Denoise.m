function img_den = DWT_2D_Denoise(img,Jmax,k,mode)
% performs hard or soft thresholding denoising of 2D signal (image)
% using scale adapted threshold
%
% To Do: implement unsupervised Bayes method
%
% Author(s): Frederic von Wegner - 08-2007
% last modified: 07/08/2007

    [nx ny] = size(img);
    nxy = nx*ny;
    [img_d,img_s] = DWT_2D_Atrous(img,Jmax);
    
    % 2D iid N(0;1) SD propagation
    wnsd = [2., 0.228, 0.117, 0.08, 0.06, 0.05];
    %wnsd = [0.891, 0.228, 0.117, 0.08, 0.06, 0.05];
    
    % estimate threshold on W1
    W1 = squeeze(img_d(:,:,1));
    W1 = reshape(W1,1,nxy);
    tW1 = median(abs(W1(:)))/0.6745;
    
    % threshold vector
    thr = k*tW1*wnsd;
    
    % start reconstruction
    img_den = img_s;
    for j=1:Jmax
        Wj = img_d(:,:,j);
        if ( strcmp(mode,'hard') )
            Wj = Wj.*(abs(Wj) > thr(j));
        end
        if ( strcmp(mode,'soft') )
            Wj = sign(Wj).*(abs(Wj)-thr(j)).*(abs(Wj) > thr(j)); 
        end        
        img_den = img_den + Wj;
    end