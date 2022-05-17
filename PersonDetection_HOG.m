%********************************************************
%*                                                      *
%*   Image Understanding Final Project                  *
%*   Install Statistics and Machine Learning Toolbo     *
%*   Install Computer Vision Toolbox                    *
%*   Date: 05/01/2022                                   *
%*                                                      *
%********************************************************

MSKPath = ["/Users/yuliu/Desktop/Clarkson/Courses/Spring_2022/CS572 Image Understanding/Assignments/Project/Person Dataset/PedMasks"];
PNGPath = ["/Users/yuliu/Desktop/Clarkson/Courses/Spring_2022/CS572 Image Understanding/Assignments/Project/Person Dataset/PNGImages"];
People_Path = ["/Users/yuliu/Desktop/Clarkson/Courses/Spring_2022/CS572 Image Understanding/Assignments/Project/Person Dataset/CRP_People"];
NoPeople_Path = ["/Users/yuliu/Desktop/Clarkson/Courses/Spring_2022/CS572 Image Understanding/Assignments/Project/Person Dataset/CRP_noPeople"];
MSKAll = dir(MSKPath);
MSKCnt= size(MSKAll,1)-2;
PNGAll=dir(PNGPath);
PNGCnt = size(PNGAll,1)-2;

%****************Extract People and NoPeople Datasets **********************
num_People=1;
num_NoPeople=1;
stride=50;
for h = 1:MSKCnt

    %********Extract People Data *************
    MSK=imread(strcat(MSKAll(h+2).folder,'/', MSKAll(h+2).name));
    testImage = imread(strcat(PNGAll(h+2).folder,'/',PNGAll(h+2).name));
    row_test= size(testImage,1);
    col_test= size(testImage,2);
    P_Cnt=max(max(MSK));
    for p=1:P_Cnt
        [x,y]=find(MSK==p);
        imcrp = testImage(min(x):max(x), min(y):max(y),:);
        imcrp = imresize(imcrp,[128 64]);
        imwrite(imcrp,strcat(People_Path,'/',"Person",num2str(num_People),'.png'));
        num_People=num_People+1;
    end

    %********Extract NoPeople Data *************
    MSK(MSK>1)=1;
    patch=ones(130,70);
    box=zeros(130,70,3);
    rowsExtra=mod(row_test,stride);
    colsExtra=mod(col_test,stride);
    imgResize = imresize(testImage,[(row_test+stride-rowsExtra) (col_test+stride-colsExtra)]);
    MSKResize = imresize(MSK,[(row_test+stride-rowsExtra) (col_test+stride-colsExtra)]);
    rowResize=size(imgResize,1);
    colResize=size(imgResize,2);
    for m=0:stride:rowResize-130
        for n=0:stride:colResize-70
            im_add=imadd(uint8(patch(1:130,1:70)), uint8(MSKResize(m+1:m+130,n+1:n+70)));
            if im_add ==1
                box=imgResize(m+1:m+130,n+1:n+70,:);
                box=imresize(box,[128 64]);
                imwrite(box,strcat(NoPeople_Path,'/',"NoPerson",num2str(num_NoPeople),'.png'));
                num_NoPeople=num_NoPeople+1;
            end
        end
    end
end
%************************************************************************


%************Create HOG Feature Vector: People and NoPeople Datasets******

[peopleFeatureVectAll, peopleCnt] = computeHOGFeature(People_Path);
[noPeopleFeatureVectAll, noPeopleCnt] = computeHOGFeature(NoPeople_Path);
dataCnt = min([peopleCnt, noPeopleCnt]);
peopleFeatureVectAll = peopleFeatureVectAll(1:dataCnt, :);
noPeopleFeatureVectAll = noPeopleFeatureVectAll(1:dataCnt, :);
allIndx = 1:1:dataCnt;
%**********************************************************************


%************************SVM Classification ****************************

%*********10 Fold Cross validation *************
crossValCnt = 10;
lastEnd = 0;
errSum = 0;
sampleSum = 0;
for i=1:crossValCnt
    startIndx = lastEnd+1;
    endIndx = round(dataCnt*(i/crossValCnt));
    
    %************Testing Data Preparation **************
    testData = peopleFeatureVectAll(startIndx:endIndx, :);
    testData = [testData; noPeopleFeatureVectAll(startIndx:endIndx, :)];
    
    %************Traning Data Preparation **************
    testIndx = startIndx:1:endIndx;
    trainIndx = setdiff(allIndx, testIndx);
    trainingData = peopleFeatureVectAll(trainIndx, :);
    trainingData = [trainingData; noPeopleFeatureVectAll(trainIndx, :)];
    
    %************Label Indicators for Training and Testing Datasets **************
    Y_train = [ones(length(trainIndx), 1); 2*ones(length(trainIndx), 1)];
    Y_test = [ones(length(testIndx), 1); 2*ones(length(testIndx), 1)];
    
    %****************SVM Modeling **********************
    SVMModel = fitcsvm(trainingData, Y_train,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto');
    [~,score] = predict(SVMModel, testData);
    result = score(:, 1) - score(:, 2);
    result(result >= 0) = 1;
    result(result ~= 1) = 2;
            
    errSum = errSum + sum(abs(result - Y_test));
    sampleSum = sampleSum + length(Y_test);
    
    lastEnd = endIndx;
end    

disp(['Average error on full dataset is ', num2str(100*(errSum/sampleSum))]);
%**************************************************************************


%**************************Sliding Window Test*****************************
trainingDataAll = peopleFeatureVectAll;
trainingDataAll = [trainingDataAll; noPeopleFeatureVectAll];
Y_trainAll = [ones(dataCnt, 1); 2*ones(dataCnt, 1)];
SVMModelTest = fitcsvm(trainingDataAll, Y_trainAll,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto');

slidingWindow = [280 200;
                 350 280];
%slidingWindow = [490 350];
winCnt = size(slidingWindow, 1);  

PeopleTestPath = ['/Users/yuliu/Desktop/Clarkson/Courses/Spring_2022/CS572 Image Understanding/Assignments/Project/Person Dataset/testPeople'];
fileAll_peopleTest = dir(PeopleTestPath);
testCnt = size(fileAll_peopleTest, 1)-3;
slideWindow_stride = 5;
for i=1:testCnt
    testImagePath = strcat(fileAll_peopleTest(i+3).folder, '/', fileAll_peopleTest(i+3).name);
    testImage = imread(testImagePath); 
    [row_testImage, col_testImage, chan_testImage] = size(testImage);
    
    figure(1), imshow(testImage)
    hold on
    
    if(chan_testImage == 3)
        imageGray = rgb2gray(testImage);
    else
        if(chan_testImage == 1)
            imageGray = testImage;
        else
            disp('ERROR!  Incorrect image format.')
        end    
    end

    for w = 1:winCnt
        %for r=1:5:(row_testImage - slidingWindow(w, 1)+1)
        for r=480:5:990    %Just for speeding up the process
            for c=1:5:(col_testImage - slidingWindow(w, 2)+1)
                imgWindow = imageGray(r:r+slidingWindow(w, 1)-1, c:c+slidingWindow(w, 2)-1);
                imgWindow = imresize(imgWindow, [128 64]);
                
                histArray_test = [];
                histArray_test = extractHOGFeatures(imgWindow);

                [~,score] = predict(SVMModelTest, histArray_test);
                if(score(1) > 0)
                    rectangle('Position',[c r slidingWindow(w, 2) slidingWindow(w, 1)],'Curvature',0.2, 'EdgeColor', 'r')
                end
            end
        end 
    end
    hold off
    wn = waitforbuttonpress;
end
%**************************************************************************


%*****Function to Compute HOG Vector for each Image in Image Folder *******

function [featureVectAll, sampleCnt] = computeHOGFeature(imgFolderPath)
    fileAll = dir(imgFolderPath);
    sampleCnt = size(fileAll, 1)-3;
    featureVectAll = zeros(sampleCnt,36*15*7);
    
    for i = 1:sampleCnt
        image = imread(strcat(fileAll(i+3).folder, '/', fileAll(i+3).name));
        histArray = [];
        histArray = extractHOGFeatures(image);
        featureVectAll(i,:)=histArray;
    end
end
%**************************************************************************
