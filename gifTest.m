f = figure
[imgs, map] = imread("homer.gif",'frames','all');
%implay(imgs)
imageObj = imshow(imgs(1:320,1:320,1,1),map)
%colormap(map)

for i = 1:10
   imageObj.CData = imgs(1:320,1:320,1,i)
   pause(0.1)
end