function [img_d,img_s] = DWT_2D_Atrous(img,Jmax)

% performs the 2D redundant ( 'à trous' ) DWT using cubic b-spline 
% scaling function
%
% ToDo:
%
% Author(s): Frederic von Wegner - 08-2007
% last modified: 07/08/2007

    
    [nx ny] = size(img);  % input image size 
    
    % mean subtraction
    mu = mean(img(:));
    img = img-mu;
    
    h1 = [1 4 6 4 1 ]/16;
    img_d = zeros(nx,ny,Jmax);
    
    % 1st decomposition level
    tmp1 = img; 
    tmp2 = img;
    tmp2 = conv2(h1,h1,tmp1,'same');
    %tmp2 = conv2fft(h1,h1,tmp1,'same');
    img_d(:,:,1) = double(tmp1) - tmp2;
    tmp1 = tmp2;
    
    if (Jmax > 1)
        for j=2:Jmax
            % update filter
            h2 = upsample(h1,2);
            h2 = h2(1:length(h2)-1);
            % update decomposition
            tmp2 = conv2(h2,h2,tmp1,'same');
            %tmp2 = conv2fft(h2,h2,tmp1,'same');
            img_d(:,:,j) = double(tmp1) - tmp2;
            tmp1 = tmp2;
            h1 = h2;        
        end
    end
    % get smooth scale
    img_s = tmp2+mu;
    