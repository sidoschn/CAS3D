%CAS2d Tool Version 1.3
%Dominik Schneidereit
%Institute of Medical Biotechnology
%2019
%5x5 pixel sobel



disp('CAS2d Tool v1.3');
disp('Institute of Medical Biotechnology, FAU Erlangen');

%set defaults
clear
wCount = 0;


defaultBasePath = 'D:\ImageDump01';
defaultFileExt = 'C00_xyz Stage Z0000.ome.tif';
%defaultBasePath = 'D:\ImageDump01\180710 CTX\180710 TN01\180710_TN01_15-27-35\testimage3';
%defaultFileExt = 'Z0_C0.ome.tif';
resultsFileName = 'CAS2dV1.3ResultsSummary';

%query path and file extension
basePath = uigetdir(defaultBasePath, 'Select base path for image detection (subfolders will also be searched)');
if basePath == 0
    exit
end
fileExt = inputdlg('Please specify a file ending to identify the first slice of a stack','File extension specification',1,{defaultFileExt});
if isempty(fileExt)
    exit
end
disp('Searching for images...');
imageFileNameList = getfn(basePath, fileExt);

%load 3d sobel kernels
load('sobel3x.mat');
load('sobel3y.mat');
load('sobel3z.mat');

%load 2d sobel kernels
load('sobel2x.mat');
load('sobel2y.mat');
load('sobel2xr2.mat');
load('sobel2yr2.mat');


%check if logfile already exists and if so, add an increasing numeral to
%the file name to keep it unique, then start a log file
if isfile([basePath '\' resultsFileName '.csv'])
    newResultsFileName = resultsFileName;
    i = 0;
    while isfile([basePath '\' newResultsFileName '.csv'])
        newResultsFileName =  [resultsFileName sprintf('%03u' ,i)];
        i=i+1;
    end
    resultsFileName = newResultsFileName;
end

resultsFileID = fopen([basePath '\' resultsFileName '.csv'],'a');
resultsHeader = [{'Filename'},{'Depth (µm)'},{'CAS (-)'},{'edgeMagnitudeWeightedCAS (-)'},{'intWeightedCAS'},{'GaussfitAmplitude (-)'},{'GaussfitSigma (°)'},{'EdgeGaussfitAmplitude (-)'},{'EdgeGaussfitSigma (°)'},{'IntGaussFitAmplitude (-)'}, {'IntGaussFitSigma (°)'}];
fprintf(resultsFileID,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',resultsHeader{:});

%checkChannelCount in first found image
disp('Checking channels...');
imageReader = bfGetReader(imageFileNameList{1});
imageMetaData = imageReader.getMetadataStore();
maxChannels =double(imageMetaData.getPixelsSizeC(0).getValue());

%query analysis channel
channelList = num2str((1:maxChannels)');
[selectedChannel, tf] = listdlg('ListString',channelList,'SelectionMode','single','PromptString','Select the channel to evaluate');

%run analysis for each detected file in FileList
waitBar1 = waitbar(0,'Analyzing Images:');
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
%     physPixUnitsX = char(imageMetaData.getPixelsPhysicalSizeX(0).unit().getSymbol());
%    physPixSizeX = imageMetaData.getPixelsPhysicalSizeX(0).getValue();
%    physPixSizeY = imageMetaData.getPixelsPhysicalSizeY(0).getValue();
%    physPixSizeZ = imageMetaData.getPixelsPhysicalSizeZ(0).getValue();
    physPixUnitsX = "micron";
    
    
    voxelHeightToWidthFactor = physPixSizeZ/physPixSizeX;
    
    %create empty matrix with dimensions of image stack
    imagePixelMatrixRaw = zeros(stackHeight,stackWidth,stackSliceCount);
    
    if (selectedChannel > stackChannelCount)
        warning('warning! selected channel exceeds existing channels! Stack is skipped!')
        wCount = wCount+1;
        continue;
    end
    
    %fill matrix with image values
    for j=1:stackSliceCount
        imagePixelMatrixRaw(:,:,j)=imageData{1,1}{(j*selectedChannel),1};
    end
    
    %free up RAM
    clear imageData;
    
    %fft(fft(X).').'
    
    minIntensity = min(imagePixelMatrixRaw,[],'all');
    maxIntensity = max(imagePixelMatrixRaw,[],'all');
    %imagePixelMatrixSmooth = imgaussfilt3(imagePixelMatrixRaw,round(size(imagePixelMatrixRaw,1))/1000);
    imagePixelMatrixSmooth = imagePixelMatrixRaw;
    for j=1:stackSliceCount
        slice = imresize(imagePixelMatrixSmooth(:,:,j),2);
        %Series{ii,1} = uint16(DWT_2D_Denoise(Series{ii,1},jMax,k,'soft'));
        %slice = DWT_2D_Denoise(imresize(imagePixelMatrixSmooth(:,:,j),2),5,5,'soft');
        %slice = DWT_2D_Denoise((imagePixelMatrixSmooth(:,:,j)),1,1,'soft');
        sliceSobelX = imfilter(slice, sobel2x);
        sliceSobelY = imfilter(slice, sobel2y);
        %sliceSobelX = imfilter(slice, sobel2xr2);
        %sliceSobelY = imfilter(slice, sobel2yr2);
%         minSliceIntensity = min(imagePixelMatrixRaw(:,:,j),[],'all');
%         maxSliceIntensity = max(imagePixelMatrixRaw(:,:,j),[],'all');
%         imshow(imagePixelMatrixRaw(:,:,j), [minSliceIntensity maxSliceIntensity]);
        %rgbSlice = cat(3, sliceSobelX, sliceSobelY);
        
        
        
        
        
        sliceVectorMag = sqrt(sliceSobelX.^2+sliceSobelY.^2);
        sliceVectorDir = atan2(sliceSobelY, sliceSobelX);
        %imshow(sliceVectorDir, [-pi pi]);
        
        redChannel = (abs(sliceSobelX).*sliceVectorMag./ max(abs(sliceSobelX),[],'all'));
        greenChannel = (abs(sliceSobelY).*sliceVectorMag./ max(abs(sliceSobelY),[],'all'));
        blueChannel = redChannel.*0;
        
        rgbSlice = cat(3, redChannel, greenChannel, blueChannel);
        %imshow(rgbSlice);
        
        sliceVectordirAng = rad2deg(sliceVectorDir);
        nbins = 360;
        
        %get the rough main direction via a histogram analysis
        [counts, edges, bin] =(histcounts(sliceVectordirAng,nbins));
        %binCenters = edges(1:end-1)+(mean(diff(edges))/2);
        binCenters = edges(1:end-1)+(mean(diff(edges)));
        [value, index] = max(counts(1:end-1));
        roughMainDirection = binCenters(index);
        
        sp0 = subplot(2,2,1);
        %plot(binCenters(1:end-1),counts(1:end-1));
        polarplot(deg2rad(binCenters),counts)
        
        singleSideDirVectorsAng = sliceVectordirAng;
        singleSideDirVectorsAng(cos(deg2rad(singleSideDirVectorsAng-roughMainDirection))<0) = NaN;
        weights = sliceVectorMag;
        weights(isnan(singleSideDirVectorsAng)) =NaN;
        %invert all vectors that are on the opposing hemicircle of the main
        %direction
        sliceVectordirAngAdj = singleSideDirVectorsAng;
        for k = 1:3
            sliceVectordirAngAdj = sliceVectordirAngAdj + (sliceVectordirAngAdj<(roughMainDirection-90)).*180;
            sliceVectordirAngAdj = sliceVectordirAngAdj - (sliceVectordirAngAdj>(roughMainDirection+90)).*180;
        end
        centeredSliceVectordirAngAdj = sliceVectordirAngAdj-roughMainDirection;
        cosAngles = cos(deg2rad(centeredSliceVectordirAngAdj));
        weightedCosAngles = cosAngles.*weights;
        nanSlice = slice;
        nanSlice(isnan(singleSideDirVectorsAng)) = NaN;
        intWeightedCosAngles = cosAngles.*nanSlice;
        edgeWeightedCAS = sum(weightedCosAngles, 'all','omitnan')/sum(weights, 'all','omitnan')
        intWeightedCAS = sum(intWeightedCosAngles, 'all','omitnan')/sum(nanSlice, 'all','omitnan')
        CAS = mean(cosAngles, 'all','omitnan')
        %   sp0 = subplot(2,2,1);
        
        sp1 = subplot(2,2,2);
        imagesc(sliceVectorMag);
        [counts, edges, bin] =(histcounts(centeredSliceVectordirAngAdj, nbins));
        %[counts, edges, bin] =(histcounts(sliceVectorDir, nbins));
        weightedCounts = zeros(1,nbins);
        for n = 1:size(weights,1)
            for m = 1:size(weights,2)
                if not(bin(n,m) == 0)
                    weightedCounts(bin(n,m)) = weightedCounts(bin(n,m))+weights(n,m);
                end
            end
        end
        intWeightedCounts = zeros(1,nbins);
        for n = 1:size(nanSlice,1)
            for m = 1:size(nanSlice,2)
                if not(bin(n,m) == 0)
                    intWeightedCounts(bin(n,m)) = intWeightedCounts(bin(n,m))+nanSlice(n,m);
                end
            end
        end
        binCenters = edges(1:end-1)+(mean(diff(edges))/2);
        %  plot(binCenters, counts);
        hold on;
        %gaussfit = fit(binCenters', counts','a1+b1*exp(-((x)/c1)^2)','StartPoint', [min(counts(2:end-1)) (max(counts(2:end-1))-min(counts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))','StartPoint', [min(counts(2:end-1)) (max(counts(2:end-1))-min(counts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))');
        %gaussAmp = gaussfit.b1;
        %gaussSigma = gaussfit.c1;
        %  plot(gaussfit);
        %plot(lorentzfit, 'b');
        %lorentzFWHM = lorentzfit.c1
        hold off;
        %  sp1 = subplot(2,2,2);
        %  imagesc(slice);
        sp3 = subplot(2,2,3);
        polarplot(deg2rad(binCenters),weightedCounts);
        %plot(binCenters, weightedCounts);
        hold on;
%        gaussfit = fit(binCenters', weightedCounts','a1+b1*exp(-((x)/c1)^2)','StartPoint', [min(weightedCounts(2:end-1)) (max(weightedCounts(2:end-1))-min(weightedCounts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))','StartPoint', [min(counts(2:end-1)) (max(counts(2:end-1))-min(counts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))');
%        gaussAmpEdgeWeighted = gaussfit.b1;
 %       gaussSigmaEdgeWeighted = gaussfit.c1;
        %  plot(gaussfit);
        hold off;
        sp4 = subplot(2,2,4);
        polarplot(deg2rad(binCenters),intWeightedCounts);
        %plot(binCenters, intWeightedCounts);
        hold on;
%        gaussfit = fit(binCenters', intWeightedCounts','a1+b1*exp(-((x)/c1)^2)','StartPoint', [min(intWeightedCounts(2:end-1)) (max(intWeightedCounts(2:end-1))-min(intWeightedCounts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))','StartPoint', [min(counts(2:end-1)) (max(counts(2:end-1))-min(counts(2:end-1))) 5], 'Robust', 'Bisquare');
        %lorentzfit = fit(binCenters', counts','a1+(2*b1/pi)*(c1/((4*x^2)+(c1^2)))');
%        gaussAmpIntWeighted = gaussfit.b1;
%        gaussSigmaIntWeighted = gaussfit.c1;
        %  plot(gaussfit);
        hold off;
        drawnow;
        %resultsHeader = [{'Filename'},{'CAS (-)'},{'edgeMagnitudeWeightedCAS (-)'},{'intWeightedCAS'},{'GaussfitAmplitude (-)'},{'GaussfitSigma (°)'},{'EdgeGaussfitAmplitude (-)'},{'EdgeGaussfitSigma (°)'},{'IntGaussFitAmplitude (-)'}, {'IntGaussFitSigma (°)'}];
        resultsList = [{imageFileNameList{i}}, {physPixSizeZ*j}, {CAS}, {edgeWeightedCAS}, {intWeightedCAS},{0},{0},{0},{0},{0},{0}];
        fprintf(resultsFileID,'%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n',resultsList{:});

        
    end
    
   
% %     
% %     
% %     % denoise with 3d gaussian filter with a sigma of 1/1000th of
% %     % image pixel number
% %     imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw,round(size(imagePixelMatrixRaw,1))/1000);
% %     %     imagePixelMatrix = imagePixelMatrixRaw;
% %     imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
% %     imagePixelMatrix = imagePixelMatrixNorm;
% %     lowerThresh = (min(imagePixelMatrix,[],'all')+0.1);
% %     upperThresh = 0.999;
% %     saturation = 100*sum((imagePixelMatrix>upperThresh), 'all')/numel(imagePixelMatrix);
% %     background = 100*sum((imagePixelMatrix<lowerThresh), 'all')/numel(imagePixelMatrix);
% %     
% %     disp(['Saturation: ' num2str(saturation) '%  Background: ' num2str(background) '%'])
% %     
% %     imagePixelMatrixRaw=imagePixelMatrix .*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
% %     
% %     %geting main direction and mean Sarcomere Length by 3d fft analysis
% %     
% %     imagefft = fftn(imagePixelMatrixRaw);
% %     imageffts = fftshift(imagefft,2);
% %     imageffts = fftshift(imageffts,1);
% %     imageffts = fftshift(imageffts,3);
% %     imagerffts = abs(real(imageffts));
% %     filtImagerffts = imgaussfilt3(imagerffts,round(size(imagerffts,1)/100));%%%%% ## this is probably tha bad boy
% %     bwThresh = max(filtImagerffts(:));
% %     iterations = 0;
% %     BWfft = imbinarize(filtImagerffts,bwThresh);
% %     stats = regionprops3(BWfft);
% %     
% %     bwThreshMod = 1.1;
% %     bwThreshModMod = 0.1;
% %     maxIterationLimit = 1000;
% %     while(~(size(stats,1)==3))
% %         bwThresh = bwThresh / bwThreshMod;
% %         BWfft = imbinarize(filtImagerffts,bwThresh);
% %         stats = regionprops3(BWfft);
% %         iterations = iterations+1;
% %         if (size(stats,1)>3)
% %             bwThresh = bwThresh*bwThreshMod;
% %             bwThreshMod = 1+((bwThreshMod-1)*bwThreshModMod);
% %         else
% %         end
% %         
% %         %in case you pass the max iteration limit, discard the image stack
% %         if iterations > maxIterationLimit
% %             break;
% %         end
% %         
% %     end
% %     
% %     if (iterations > maxIterationLimit)
% %         warning('warning! Unable to detect secondary peaks in fourier Transform! Stack is skipped!')
% %         wCount = wCount+1;
% %         continue;
% %     end
% %     
% %     %getting centroid vectors of secondary peaks by picking one of the
% %     %longer vectors (major peak centroid vector length should be zero)
% %     
% %     xVectors = stats.Centroid(:,1)-((size(imagefft,1)/2)+1);
% %     yVectors = stats.Centroid(:,2)-((size(imagefft,2)/2)+1);
% %     %zVectors = (stats.Centroid(:,3)-((size(imagefft,3)/2))-1)*voxelHeightToWidthFactor;
% %     %ratio is unneccessary due to how FFT works and is removed 
% %     zVectors = stats.Centroid(:,3)-((size(imagefft,3)/2))-1;
% %     
% %     magnitudefftVectors = sqrt((xVectors.^2)+(yVectors.^2)+(zVectors.^2));
% %     
% %     if magnitudefftVectors(1)>magnitudefftVectors(2)
% %         xVector = xVectors(1);
% %         yVector = yVectors(1);
% %         zVector = zVectors(1);
% %         
% %     else
% %         xVector = xVectors(2);
% %         yVector = yVectors(2);
% %         zVector = zVectors(2);
% %         
% %     end
% %     mainDirection = atan2(yVector,xVector);
% %     mainElevation = atan2(sqrt((xVector.^2)+(yVector.^2)),zVector);
% %     mainDirectionDeg = rad2deg(atan2(yVector,xVector));
% %     mainElevationDeg = rad2deg(atan2(sqrt((xVector.^2)+(yVector.^2)),zVector));
% %     magnitudefftVector = sqrt((xVector.^2)+(yVector.^2)+(zVector.^2));
% %     meanSL = physPixSizeX*size(imagefft,1)/magnitudefftVector;
% %     
% %     disp(['Iterations neccessary to find secondary peaks: ' num2str(iterations)]);
% %     disp(['Detected main direction ' num2str(mainDirectionDeg) '° and elevation ' num2str(mainElevationDeg) '°' ' with mean SL: ' num2str(meanSL) ' ' physPixUnitsX]);
% %     
% %     % create norm vector with main direction and elevation heading
% %     xyzMainVector = zeros(1,3);
% %     xyzMainVector(1) = sin(mainElevation)*cos(mainDirection); %vector x
% %     xyzMainVector(2) = sin(mainElevation)*sin(mainDirection); %vector y
% %     xyzMainVector(3) = cos(mainElevation); %vector z
% %     
% % %     % denoise with 3d gaussian filter with a sigma of 1/1000th of
% % %     % image pixel number
% % %     
% % %     imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw,round(size(imagePixelMatrixRaw,1))/1000);
% % %     %     imagePixelMatrix = imagePixelMatrixRaw;
% % %     imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
% % %     imagePixelMatrix = imagePixelMatrixNorm;
% % %     lowerThresh = (min(imagePixelMatrix,[],'all')+0.1);
% % %     upperThresh = 0.999;
% % %     saturation = 100*sum((imagePixelMatrix>upperThresh), 'all')/numel(imagePixelMatrix);
% % %     background = 100*sum((imagePixelMatrix<lowerThresh), 'all')/numel(imagePixelMatrix);
% % %     
% % %     disp(['Saturation: ' num2str(saturation) '%  Background: ' num2str(background) '%'])
% % %     
% %     %create sobel vectors for pixels within the threshold limits
% %     imageSobelX = imfilter(imagePixelMatrix,sobel3x,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
% %     imageSobelY = imfilter(imagePixelMatrix,sobel3z,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
% %     imageSobelZ = imfilter(imagePixelMatrix,sobel3y,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)))./voxelHeightToWidthFactor;
% %     
% %     %get Magnitudes of sobel vectors and set zero length vectors to NaN
% %     %in order to exclude them from evaluation
% %     sobelMagnitudes = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
% %     sobelMagnitudes(sobelMagnitudes==0) = NaN;
% %     %totalSobelVectorLength = sum(sobelMagnitudes, 'all');
% %     
% %     %normalize sobel vectors to a length of 1 (to enable a unweighted
% %     %determination of directionalities)
% %     
% %     imageSobelX = imageSobelX./sobelMagnitudes;
% %     imageSobelY = imageSobelY./sobelMagnitudes;
% %     imageSobelZ = imageSobelZ./sobelMagnitudes;
% %     
% %     %-----
% %     %here directionality and magnitude of each pixel is determined,
% %     %-----
% %     
% %     %create empty matrix to hold normal vectors
% %     imageVectorMatrix = zeros(stackHeight,stackWidth,stackSliceCount,4);
% %     
% %     %calculate direction angle
% %     imageVectorMatrix(:,:,:,1) = rad2deg(atan2(imageSobelY,imageSobelX));
% %     
% %     %calculate magnitude
% %     imageVectorMatrix(:,:,:,3) = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
% %     
% %     %calculate elevation angle
% %     imageVectorMatrix(:,:,:,2) = rad2deg(atan2(sqrt((imageSobelX.^2)+(imageSobelY.^2)),imageSobelZ));
% %     
% %     %calculate cosines of divergence angle from main direction
% %     imageVectorMatrix(:,:,:,4) = abs(imageSobelX.*xyzMainVector(1)+imageSobelY.*xyzMainVector(2)+imageSobelZ.*xyzMainVector(3));
% %     
% %     %calculate the cosine angle sum of the image stack
% %     CAS3d = nanmean(imageVectorMatrix(:,:,:,4),'all');
% %     
% %     %calculate the intensity weighted cosine angle sum of the image stack
% %     CAS3dIntWeight = nansum((imageVectorMatrix(:,:,:,4).*imagePixelMatrix),'all')/nansum((((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)).*imagePixelMatrix),'all');
% %     
% %     disp(['CAS3d = ' num2str(CAS3d) ' wCAS3d = ' num2str(CAS3dIntWeight)]);
% %     
% %     %----
% %     %here the output plot is generated and saved
% %     %----
% %     
% %     %subplot 1 shows the intensity weighted histogram of divergence from
% %     %the main direction in °
% %     sp0 = subplot(2,2,1);
% %     [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))).*imagePixelMatrix);
% %     findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
% %     [peakInts, weightedPeakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
% %     if (numel(weightedPeakLocs)==0)
% %         weightedPeakLocs = [NaN NaN NaN];
% %     end
% %     displayText0 = {[' Major Divergence: ' num2str(weightedPeakLocs(1)) '°']};
% %     if exist('displayText0item')
% %         delete(displayText0item);
% %     end
% %     displayText0item = text(weightedPeakLocs(1),peakInts(1),displayText0);
% %     xlabel('divergence angle (°)')
% %     ylabel('binned signal intensity (-)')
% %     
% %     %subplot 2 shows the histogram of divergence from main direction in °
% %     sp1 = subplot(2,2,2);
% %     [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))));
% %     findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
% %     [peakInts, peakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
% %     if (numel(peakLocs)==0)
% %         peakLocs = [NaN NaN NaN];
% %         peakInts = [NaN NaN NaN];
% %     end
% %     displayText1 = {[' Major Divergence: ' num2str(peakLocs(1)) '°']};
% %     if exist('displayText1item')
% %         delete(displayText1item);
% %     end
% %     displayText1item = text(peakLocs(1),peakInts(1),displayText1);
% %     xlabel('divergence angle (°)')
% %     ylabel('binned pixel counts (-)')
% %     
% %     %subplot 3 shows the center slice of the image stack with the cosine
% %     %angle divergence from the main direciton of each pixel color coded
% %     %from 0-1. Pixels that are not used in the evaluation are set to 0
% %     sp2 = subplot(2,2,3);
% %     imagesc(imageVectorMatrix(:,:,round(stackSliceCount/2),4));
% %     axis off;
% %     title('cosines of center slice');
% %     colorbar('eastoutside');
% %     
% %     %subplot 4 displays a summary of the detected parameters
% %     sp3 = subplot(2,2,4);
% %     cla(sp3)
% %     axis off;
% %     displayText = {['Main direction: ' num2str(mainDirectionDeg) '°'],['Main elevation: ' num2str(mainElevationDeg) '°'],['CAS3d: ' num2str(CAS3d)],['Weighted CAS3d: ' num2str(CAS3dIntWeight)],['Saturation: ' num2str(saturation) '%'], ['mean SL: ' num2str(meanSL) ' ' physPixUnitsX]};
% %     if exist('textItem')
% %         delete(textItem);
% %     end
% %     %textItem = text(0,0.5,displayText);
% %     
% %     %the plot is saved in the base folder with the name of the image stack
% %     %or first slice of the image stack
% %     print([basePath '\' ImageFileName '.dirDist.png'],'-dpng','-r400');
% %     
% %     %the results are written to the results list array and saved to file
% %     resultsList = [{imageFileNameList{i}}, {CAS3d}, {CAS3dIntWeight}, {mainDirectionDeg},{mainElevationDeg},{weightedPeakLocs(1)},{peakLocs(1)},{saturation},{background},{meanSL}];
% %     %fprintf(resultsFileID,'%s;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f\n',resultsList{:});
% %     
end

%perform clean up
fclose(resultsFileID);
close(waitBar1);

%display end message
disp('Analysis complete!');
if wCount>0
    disp(['With ' num2str(wCount) ' skipped stacks due to warnings']);
end
disp(['Find your log file and figures at ' basePath]);