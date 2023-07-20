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

resultsList = [{'Filename'},{'CAS3d'},{'MainDirection'},{'MainElevation'},{'WeightedAngularDeviation'},{'AngularDeviation'}];

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

waitBar1 = waitbar(0,'Analyzing Images:')

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
    imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw,1);
%     imagePixelMatrix = imagePixelMatrixRaw;
    imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
    imagePixelMatrix = imagePixelMatrixNorm;
    lowerThresh = 0.1;
    upperThresh = 0.99;
    
    imageSobelX = imfilter(imagePixelMatrix,sobel3x,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelY = imfilter(imagePixelMatrix,sobel3z,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelZ = imfilter(imagePixelMatrix,sobel3y,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)))./voxelHeightToWidthFactor;
    
    sobelMagnitudes = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    sobelMagnitudes(sobelMagnitudes==0) = NaN;
    
    normSobelX = imageSobelX./sobelMagnitudes;
    normSobelY = imageSobelY./sobelMagnitudes;
    normSobelZ = imageSobelZ./sobelMagnitudes;
    
    imageSobelX = normSobelX;
    imageSobelY = normSobelY;
    imageSobelZ = normSobelZ;
    
    imageSobel = cat(4,imageSobelX, imageSobelY, imageSobelZ);
    
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
    %imageVectorMatrix(:,:,:,3) = permute(sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2)),[2 1 3 4]);
    imageVectorMatrix(:,:,:,3) = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    
    %calculate elevation angle
    
    %imageVectorMatrix(:,:,:,2) = permute(rad2deg(atan2(sqrt((imageSobelX.^2)+(imageSobelY.^2)),imageSobelZ)),[2 1 3 4]);
    imageVectorMatrix(:,:,:,2) = rad2deg(atan2(sqrt((imageSobelX.^2)+(imageSobelY.^2)),imageSobelZ));
    
    %imageVectorMatrix(:,:,:,2) = rad2deg(acos(imageSobelZ./imageVectorMatrix(:,:,:,3)))-90;
    
    
    %normalize magnitude
    %imageVectorMatrix(:,:,:,3) = imageVectorMatrix(:,:,:,3)./max(max(max(imageVectorMatrix(:,:,:,3))));
  
    %---------
    % here main direction and elevation are determined
    %---------
    
    %check the overall vector magnitude distribution and only use vectors
    %with a relevant magnitude for mean directionality determination
    
%     relevanceWidth = 1;
%     [histData, histEdges] = histcounts(imageVectorMatrix(:,:,:,3));
%     logHistEdges = log(histEdges(2:end));
%     gaussfit = fit(logHistEdges.', histData.','gauss1');
%     fitCoefficients = coeffvalues(gaussfit);
%     lowerThresh = exp(fitCoefficients(2)+(fitCoefficients(3)*1))
%     %lowerThresh = exp(fitCoefficients(2))
%     upperThresh = exp(fitCoefficients(2)+(fitCoefficients(3)*3*relevanceWidth))

    
%     lowerThresh = mean(imageVectorMatrix(:,:,:,3),'all');
   


    %creating sobel sets with only positive and only negative vector
    %components, keep only vector components with a magnitude of at least
    %0.5
    
    
    %imageSobelXRelev = imageSobelX.*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelXpos = imageSobelX;
    imageSobelXneg = imageSobelX;
%     imageSobelXpos = imageSobelX;
%     imageSobelXneg = imageSobelX;
    imageSobelXpos(imageSobelXpos<0) = [];
    imageSobelXneg(imageSobelXneg>0) = [];
    
    %imageSobelYRelev = imageSobelY.*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    
    imageSobelYpos = imageSobelY;
    imageSobelYneg = imageSobelY;
%     imageSobelYpos = imageSobelY;
%     imageSobelYneg = imageSobelY;
    imageSobelYpos(imageSobelYpos<0) = [];
    imageSobelYneg(imageSobelYneg>0) = [];
    
    %imageSobelZRelev = imageSobelZ.*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
   
    imageSobelZpos = imageSobelZ;
    imageSobelZneg = imageSobelZ;
%     imageSobelZpos = imageSobelZ;
%     imageSobelZneg = imageSobelZ;
    imageSobelZpos(imageSobelZpos<0) = [];
    imageSobelZneg(imageSobelZneg>0) = [];
    
    %determining main direction in positive and negative directions
    %and 
    
%     sumOfXvektorsAll = sum(imageSobelXRelev,'all');
%     sumOfYvektorsAll = sum(imageSobelYRelev,'all');
%     sumOfZvektorsAll = sum(imageSobelZRelev,'all');
    
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
    
%     mainDirectionAll = (atan2(sumOfYvektorsAll,sumOfXvektorsAll));
%     mainElevationAll = (atan2(sqrt((sumOfXvektorsAll.^2)+(sumOfYvektorsAll.^2)),sumOfZvektorsAll));
%     
%     mainDirectionAllDeg = rad2deg(mainDirectionAll)
%     mainElevationAllDeg = rad2deg(mainElevationAll)
    
    %create norm vector with main direction and elevation heading
    
    xyzMainVector = zeros(1,3);
    xyzMainVector(1) = sin(mainElevation)*cos(mainDirection); %vector x
    xyzMainVector(2) = sin(mainElevation)*sin(mainDirection); %vector y
    xyzMainVector(3) = cos(mainElevation); %vector z
   
    
    %a Histogram with 1° bin size is created for visualization purposes
  
    %binnedHistogram = zeros(181,361,3);
    %binnedHistogram = zeros(1001,1001,3);
    cosAngleDivergencePre1 = imageSobelX.*xyzMainVector(1);
    cosAngleDivergencePre2 = imageSobelY.*xyzMainVector(2);
    cosAngleDivergencePre3 = imageSobelZ.*xyzMainVector(3);
    cosAngleDivergencePre4 = cosAngleDivergencePre1+cosAngleDivergencePre2+cosAngleDivergencePre3;
    imageVectorMatrix(:,:,:,4) = abs(cosAngleDivergencePre4./(imageVectorMatrix(:,:,:,3).*norm(xyzMainVector)));
%     cosAngleDivergence = abs(cosAngleDivergencePre4./(sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2))*norm(xyzMainVector)));
    
%     for x=1:size(imageVectorMatrix,1)
%         for y=1:size(imageVectorMatrix,2)
%             for z=1:size(imageVectorMatrix,3)
% %                 direc = imageVectorMatrix(x,y,z,1);
% %                 dirBin = round(1000*(sin(deg2rad(direc)).^2))+1;
% %                 elev = imageVectorMatrix(x,y,z,2);
% %                 elevBin = round(1000*(sin(deg2rad(elev)).^2))+1;
% %                 magnitude = imageVectorMatrix(x,y,z,3);
%                 if ((~(sobelMagnitudes(x,y,z)==0))&&(~isnan(sobelMagnitudes(x,y,z))))
%                     sobelVektor = [imageSobelX(x,y,z), imageSobelY(x,y,z), imageSobelZ(x,y,z)];
%                     %cosAngleDivergence2 = cos(atan2(norm(cross(xyzMainVector,sobelVektor)), dot(xyzMainVector,sobelVektor)));
%                     cosAngleDivergence = abs(dot(xyzMainVector,sobelVektor)/(norm(sobelVektor)*norm(xyzMainVector)));
%                     imageVectorMatrix(x,y,z,4) = cosAngleDivergence;
%                     %imageVectorMatrix(x,y,z,5) = cosAngleDivergence*magnitude;
%                     %imageVectorMatrix(x,y,z,6) = cosAngleDivergence2;
%                 else
%                    imageVectorMatrix(x,y,z,4) = NaN;
%                    %imageVectorMatrix(x,y,z,5) = NaN;
%                    imageVectorMatrix(x,y,z,3) = NaN;
%                    %imageVectorMatrix(x,y,z,6) = NaN;
%                 end
%                 
%                 
% %                 if(isnan(dirBin))
% %                 else
% %                     binnedHistogram(elevBin,dirBin,1) = binnedHistogram(elevBin,dirBin,1)+ magnitude;
% %                     binnedHistogram(elevBin,dirBin,2) = binnedHistogram(elevBin,dirBin,2)+ 1; 
% %                     binnedHistogram(elevBin,dirBin,3) = binnedHistogram(elevBin,dirBin,3)+ imagePixelMatrix(x,y,z);
% %                 end
%                 
%                 
%             end
%         end
%     end
    
    
    
%     %filteredMagnitudeImage = imgaussfilt(binnedHistogram(:,:,1),1);
%     filteredMagnitudeImage = binnedHistogram(:,:,1);
%     maximaRegions = imextendedmax(filteredMagnitudeImage,mean(filteredMagnitudeImage,'all'));
%     
%     stats = regionprops(maximaRegions,filteredMagnitudeImage,'WeightedCentroid');
    %sp1 = subplot(2,2,[1,2]);
    sp0 = subplot(2,2,1);
    [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))).*imagePixelMatrix);
    findpeaks(smoothdata(divY,'gaussian'),divX(1:end-1))
    [peakInts, weightedPeakLocs] = findpeaks(smoothdata(divY,'gaussian'),divX(1:end-1));
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
    findpeaks(smoothdata(divY,'gaussian'),divX(1:end-1))
    [peakInts, peakLocs] = findpeaks(smoothdata(divY,'gaussian'),divX(1:end-1));
    if (numel(peakLocs)==0)
        peakLocs = [NaN NaN NaN];
    end
    displayText1 = {[' Major Divergence: ' num2str(peakLocs(1)) '°']};
    if exist('displayText1item')
        delete(displayText1item);
    end
    displayText1item = text(peakLocs(1),peakInts(1),displayText1);
    xlabel('divergence angle (°)') 
    ylabel('binned pixel counts (-)') 
    
    CAS3d = nanmean(imageVectorMatrix(:,:,:,4),'all');
    %CAS3d2 = nanmean(imageVectorMatrix(:,:,:,6),'all');
    %weightedCAS3d = nansum(imageVectorMatrix(:,:,:,5),'all')/nansum(imageVectorMatrix(:,:,:,3),'all');
    %weightedCAS3dIntensity = nansum(imageVectorMatrix(:,:,:,6),'all')/nansum(imagePixelMatrix,'all');
    disp(['CAS3d = ' num2str(CAS3d)]);
%     weightedADM = rad2deg(acos(imageVectorMatrix(:,:,:,4)));%.*imagePixelMatrix;
%     [divY divX] = histcounts(weightedADM);
%     divXmirr = [(-1*fliplr(divX(2:end))) divX(2:end)];
%     divYmirr = [fliplr(divY) divY];
%     [fitData fitGOF] = fit(divXmirr.',divYmirr.','gauss1');
%     fitCoeffs = coeffvalues(fitData);
%     plot(fitData,divXmirr,divYmirr);
%     xlabel('intensity');
%     ylabel('divergence angle'); 
%     fitDisplayText = {['Angular Divergence (StDev): ' num2str(fitCoeffs(3)) '°'],['Fit R²: ' num2str(fitGOF.rsquare)]};
%     if exist('fitText')
%         delete(textItem);
%     end
%     fitText = text(20,fitCoeffs(1),fitDisplayText);
%     binnedDistribFigure = imagesc(binnedHistogram(:,:,1));
%     xlabel('xy-direction (°)') 
%     ylabel('z-elevation(°)') 
    sp2 = subplot(2,2,3);
    imagesc(imageVectorMatrix(:,:,round(stackSliceCount/2),4));
    axis off;
    title('cosines of center slice');
    colorbar('eastoutside');
    sp3 = subplot(2,2,4);
    cla(sp3)
    axis off;
    displayText = {['Main direction: ' num2str(mainDirectionDeg) '°'],['Main elevation: ' num2str(mainElevationDeg) '°'],['CAS3d: ' num2str(CAS3d)]};
    if exist('textItem')
        delete(textItem);
    end
    
    textItem = text(0,0.5,displayText);
    
    print([basePath '\' ImageFileName '.dirDist.png'],'-dpng','-r400');
    %saveas(binnedDistribFigure,[basePath '\' ImageFileName '.dirDist.png'],'png');
%     hold on;
%     for s=1:numel(stats)
%         
%     end
%     plot(stats(1).WeightedCentroid(1),stats(1).WeightedCentroid(2),'r*','MarkerSize', 15)
%     
%     hold off;
    
    %hhhs = [0;0;0];
    
    
    resultsList = vertcat(resultsList, [{imageFileNameList{i}}, {CAS3d}, {mainDirectionDeg},{mainElevationDeg},{weightedPeakLocs(1)},{peakLocs(1)}]);
    
    
%     
%     xvector1 = reshape(imageVectorMatrix(:,:,:,2),[numel(imageVectorMatrix(:,:,:,2)) 1]);
%     zvector1 = reshape(imageVectorMatrix(:,:,:,3),[numel(imageVectorMatrix(:,:,:,2)) 1]);
%    
%     
%     yvector1 = reshape(imageVectorMatrix(:,:,:,1),[numel(imageVectorMatrix(:,:,:,2)) 1]);
%     scatter(xvector1,yvector1,zvector1)
    
end


cell2csv([basePath '\CAS3dResultsSummary.xls'],resultsList);
close(waitBar1);
disp('Analysis complete!');