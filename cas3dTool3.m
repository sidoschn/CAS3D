clear
defaultBasePath = 'D:\';
%defaultBasePath = 'D:\ImageDump01\180208 IO PreDigestion\180208_IO1Stack2_14-40-34';

defaultFileExt = 'C00_xyz Stage Z0000.ome.tif';
% defaultFileExt = 'C00_z0000.ome.tif';
%defaultFileExt = '.tif';

basePath = uigetdir(defaultBasePath, 'Select base path for image detection');
fileExt = inputdlg('Please specify file ending to identify first slice of stack','File extension specification',1,{defaultFileExt});
disp('Searching for images...');
imageFileNameList = getfn(basePath, fileExt);       %Dateien im Ordner und unterordner erfassen

load('sobel3x.mat');
load('sobel3y.mat');
load('sobel3z.mat');

maxChannels = 0;

resultsList = [{'Filename'},{'CAS3d (-)'},{'intWeightedCAS3d (-)'},{'MainDirection (°)'},{'MainElevation (°)'},{'WeightedAngularDeviation (°)'},{'AngularDeviation (°)'},{'Saturation (%)'},{'Background (%)'}];

%checkChannelCount
disp('Checking channels...');
%for i=1:numel(imageFileNameList)
    imageReader = bfGetReader(imageFileNameList{1});
    imageMetaData = imageReader.getMetadataStore();
    %stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    maxChannels = max([maxChannels double(imageMetaData.getPixelsSizeC(0).getValue())]);
%end

channelList = num2str((1:maxChannels)');

[selectedChannel, tf] = listdlg('ListString',channelList,'SelectionMode','single','PromptString','Select the channel to evaluate');

waitBar1 = waitbar(0,'Analyzing Images:');

%run analysis for each detected file in FileList
for i=1:numel(imageFileNameList)
    waitbar((i/numel(imageFileNameList)),waitBar1,['Analyzing Images:' num2str(i) '/' num2str(numel(imageFileNameList))]);
    [folderPath, ImageFileName, ImageFileExtension] = fileparts(imageFileNameList{i});
    disp(['Processing: ' ImageFileName])
    imageData = bfopen(imageFileNameList{i});
    imageMetaData = imageData{1,4};
    stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    stackSliceCount = imageMetaData.getPixelsSizeZ(0).getValue();
    stackWidth = imageMetaData.getPixelsSizeX(0).getValue();
    stackHeight = imageMetaData.getPixelsSizeY(0).getValue();
    physPixSizeX = imageMetaData.getPixelsPhysicalSizeX(0).value().doubleValue();
    physPixSizeY = imageMetaData.getPixelsPhysicalSizeY(0).value().doubleValue();
    physPixSizeZ = imageMetaData.getPixelsPhysicalSizeZ(0).value().doubleValue();
    
    voxelHeightToWidthFactor = physPixSizeZ/physPixSizeX;
%     voxelHeightToWidthFactor = 1;
    
    %create empty matrix with dimensions of image stack
    imagePixelMatrixRaw = zeros(stackHeight,stackWidth,stackSliceCount);
    
    if (selectedChannel > stackChannelCount)
        warning('warning! selected channel exceeds existing channels! Stack is skipped!')
    continue;
    end
    
    %fill matrix with image values
    for j=1:stackSliceCount
        imagePixelMatrixRaw(:,:,j)=imageData{1,1}{(j*selectedChannel),1};
    end
    clear imageData;
    % denoise with 3d gaussian filter 
    imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw);
    %imagePixelMatrix = imagePixelMatrixRaw;
    imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
    imagePixelMatrix = imagePixelMatrixNorm;
    lowerThresh = 0.1;
    upperThresh = 0.999;
    saturation = 100*sum((imagePixelMatrix>upperThresh), 'all')/numel(imagePixelMatrix);
    background = 100*sum((imagePixelMatrix<lowerThresh), 'all')/numel(imagePixelMatrix);
    
    disp(['Saturation: ' num2str(saturation) '%  Background: ' num2str(background) '%'])
    %create sobel vectors for pixels within the threshold limits
    imageSobelX = imfilter(imagePixelMatrix,sobel3x,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelY = imfilter(imagePixelMatrix,sobel3z,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelZ = imfilter(imagePixelMatrix,sobel3y,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)))./voxelHeightToWidthFactor;
    
    sobelMagnitudes = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    sobelMagnitudes(sobelMagnitudes==0) = NaN;
    
    %normalize sobel vectors to a length of 1 (to enable a unweighted
    %determination of directionalities)
    
    imageSobelX = imageSobelX./sobelMagnitudes;
    imageSobelY = imageSobelY./sobelMagnitudes;
    imageSobelZ = imageSobelZ./sobelMagnitudes;
    
%     normSobelX = imageSobelX./sobelMagnitudes;
%     normSobelY = imageSobelY./sobelMagnitudes;
%     normSobelZ = imageSobelZ./sobelMagnitudes;
%     
%     imageSobelX = normSobelX;
%     imageSobelY = normSobelY;
%     imageSobelZ = normSobelZ;
    
    %imageSobel = cat(4,imageSobelX, imageSobelY, imageSobelZ);
    
     %-----
    %here directionality and magnitude of each pixel is determined, 
    
    %-----
    
    %create empty matrix to hold normal vectors
    imageVectorMatrix = zeros(stackHeight,stackWidth,stackSliceCount,4);
    
    %calculate direction angle
    
    %imageVectorMatrix(:,:,:,1) = permute(rad2deg(atan2(imageSobelY,imageSobelX)),[2 1 3 4]);
    imageVectorMatrix(:,:,:,1) = rad2deg(atan2(imageSobelY,imageSobelX));
    
    %imageVectorMatrix(:,:,:,1) = rad2deg(acos(imageSobelX./(sqrt((imageSobelX.^2)+(imageSobelY.^2)))));
    
    %calculate magnitude
    imageVectorMatrix(:,:,:,3) = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    
    %calculate elevation angle
    imageVectorMatrix(:,:,:,2) = rad2deg(atan2(sqrt((imageSobelX.^2)+(imageSobelY.^2)),imageSobelZ));
    
    
    
 
    %---------
    % here main direction and elevation are determined
    %---------

    %creating sobel sets with only positive and only negative vector
    %components

    %intensity weighting was performed here, as it seems to increase the
    %accuracy of the determined main direction
    
    imageSobelXpos = imageSobelX.*imagePixelMatrix;
    imageSobelXneg = imageSobelX.*imagePixelMatrix;
%     imageSobelXpos = imageSobelX;
%     imageSobelXneg = imageSobelX;
    imageSobelXpos(imageSobelXpos<0) = [];
    imageSobelXneg(imageSobelXneg>0) = [];
        
    imageSobelYpos = imageSobelY.*imagePixelMatrix;
    imageSobelYneg = imageSobelY.*imagePixelMatrix;
%     imageSobelYpos = imageSobelY;
%     imageSobelYneg = imageSobelY;
    imageSobelYpos(imageSobelYpos<0) = [];
    imageSobelYneg(imageSobelYneg>0) = [];
       
    imageSobelZpos = imageSobelZ.*imagePixelMatrix;
    imageSobelZneg = imageSobelZ.*imagePixelMatrix;
%     imageSobelZpos = imageSobelZ;
%     imageSobelZneg = imageSobelZ;
    imageSobelZpos(imageSobelZpos<0) = [];
    imageSobelZneg(imageSobelZneg>0) = [];
    
    %determining main direction in positive and negative directions
    %and 

    sumOfXvektors = nansum(imageSobelXpos,'all');
    sumOfYvektors = nansum(imageSobelYpos,'all');
    sumOfZvektors = nansum(imageSobelZpos,'all');
    sumOfXvektorsNeg = nansum(imageSobelXneg,'all');
    sumOfYvektorsNeg = nansum(imageSobelYneg,'all');
    sumOfZvektorsNeg = nansum(imageSobelZneg,'all');
    
    if (sumOfYvektors == 0 && sumOfYvektorsNeg == 0 && sumOfXvektors == 0 && sumOfXvektorsNeg == 0)
        % this is to handle the special case of perfectly homogeneous data
        % in the x-y plane (which should only occur in test scenarios
        % anyways) and sets the main direction to 0° in this case
        mainDirectionPos = 1;
        mainDirectionNeg = 1;
    else
        mainDirectionPos = cos(atan2(sumOfYvektors,sumOfXvektors));
        mainDirectionNeg = cos(atan2(sumOfYvektorsNeg,sumOfXvektorsNeg)+pi);
    end
  
    %getting main Elevation and Direction in degrees, as well as a measure
    %for symmetry
    mainElevationPos = cos(atan2(sqrt((sumOfXvektors.^2)+(sumOfYvektors.^2)),sumOfZvektors));
    mainElevationNeg = cos(atan2(sqrt((sumOfXvektorsNeg.^2)+(sumOfYvektorsNeg.^2)),sumOfZvektorsNeg)+pi);
    deltaDirPlusMinus = abs(rad2deg(acos(mainDirectionPos))-rad2deg(acos(mainDirectionNeg)));
    deltaElevPlusMinus = abs(rad2deg(acos(mainElevationPos))-rad2deg(acos(mainElevationNeg)));
    mainDirection = acos(mean([mainDirectionPos mainDirectionNeg]));
    mainElevation = acos(mean([mainElevationPos mainElevationNeg]));
    mainDirectionDeg = rad2deg(mainDirection);
    mainElevationDeg = rad2deg(mainElevation);
    
    disp(['Detected main direction ' num2str(mainDirectionDeg) '° and elevation ' num2str(mainElevationDeg) '°']);
    
    %create norm vector with main direction and elevation heading
    
    xyzMainVector = zeros(1,3);
    xyzMainVector(1) = sin(mainElevation)*cos(mainDirection); %vector x
    xyzMainVector(2) = sin(mainElevation)*sin(mainDirection); %vector y
    xyzMainVector(3) = cos(mainElevation); %vector z
 
%     cosAngleDivergencePre1 = imageSobelX.*xyzMainVector(1);
%     cosAngleDivergencePre2 = imageSobelY.*xyzMainVector(2);
%     cosAngleDivergencePre3 = imageSobelZ.*xyzMainVector(3);
    imageVectorMatrix(:,:,:,4) = abs(imageSobelX.*xyzMainVector(1)+imageSobelY.*xyzMainVector(2)+imageSobelZ.*xyzMainVector(3));
%     cosAngleDivergencePre4 = cosAngleDivergencePre1+cosAngleDivergencePre2+cosAngleDivergencePre3;
%     imageVectorMatrix(:,:,:,4) = abs(cosAngleDivergencePre4./(norm(xyzMainVector)));
   % imageVectorMatrix(:,:,:,4) = abs(cosAngleDivergencePre4./(imageVectorMatrix(:,:,:,3).*norm(xyzMainVector)));
%     cosAngleDivergence = abs(cosAngleDivergencePre4./(sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2))*norm(xyzMainVector)));
   
    sp0 = subplot(2,2,1);
    [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))).*imagePixelMatrix);
    findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
    [peakInts, weightedPeakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
    if (numel(weightedPeakLocs)==0)
        weightedPeakLocs = [NaN NaN NaN];
    end
    displayText0 = {[' Major Divergence: ' num2str(weightedPeakLocs(1)) '°']};
    if exist('displayText0item')
        delete(displayText0item);
    end
    displayText0item = text(weightedPeakLocs(1),peakInts(1),displayText0);
    xlabel('divergence angle (°)') 
    ylabel('binned signal intensity (-)') 
    
    sp1 = subplot(2,2,2);
    [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))));
    findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
    [peakInts, peakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
    if (numel(peakLocs)==0)
        peakLocs = [NaN NaN NaN];
        peakInts = [NaN NaN NaN];
    end
    displayText1 = {[' Major Divergence: ' num2str(peakLocs(1)) '°']};
    if exist('displayText1item')
        delete(displayText1item);
    end
    displayText1item = text(peakLocs(1),peakInts(1),displayText1);
    xlabel('divergence angle (°)') 
    ylabel('binned pixel counts (-)') 
    
    CAS3d = nanmean(imageVectorMatrix(:,:,:,4),'all');
    CAS3dIntWeight = nansum((imageVectorMatrix(:,:,:,4).*imagePixelMatrix),'all')/nansum((((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)).*imagePixelMatrix),'all');
   
    disp(['CAS3d = ' num2str(CAS3d) ' wCAS3d = ' num2str(CAS3dIntWeight)]);

    sp2 = subplot(2,2,3);
    imagesc(imageVectorMatrix(:,:,round(stackSliceCount/2),4));
    axis off;
    title('cosines of center slice');
    colorbar('eastoutside');
    sp3 = subplot(2,2,4);
    cla(sp3)
    axis off;
    displayText = {['Main direction: ' num2str(mainDirectionDeg) '°'],['Main elevation: ' num2str(mainElevationDeg) '°'],['CAS3d: ' num2str(CAS3d)],['Weighted CAS3d: ' num2str(CAS3dIntWeight)],['Saturation: ' num2str(saturation) '%']};
    if exist('textItem')
        delete(textItem);
    end
    textItem = text(0,0.5,displayText);
    
    print([basePath '\' ImageFileName '.dirDist.png'],'-dpng','-r400');
    resultsList = vertcat(resultsList, [{imageFileNameList{i}}, {CAS3d}, {CAS3dIntWeight}, {mainDirectionDeg},{mainElevationDeg},{weightedPeakLocs(1)},{peakLocs(1)},{saturation},{background}]);
        
end


cell2csv([basePath '\CAS3dResultsSummary.xls'],resultsList);
close(waitBar1);
disp('Analysis complete!');