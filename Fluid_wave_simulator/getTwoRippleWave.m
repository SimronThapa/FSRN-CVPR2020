%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
function [warp, levels] = getTwoRippleWave(img, nFrame, alpha, c, R, R1, k)

if size(img, 1) > 1 && size(img, 2) > 1
    h = size(img, 1);
    w = size(img, 2);
else
    w = img(1);
    h = img(2);
end

gKernel = fspecial('gaussian', [81, 81], 10);
lap = fspecial('laplacian');
soby = fspecial('sobel');
sobx = soby';

% generate random h
% [X, Y] = meshgrid(1:w, 1:h);
% di = X - floor(w/2);
% dj = Y - floor(h/2);
% sigma = 30;
% waterLevel = exp(-(di.^2 + dj.^2) / 2 / sigma^2);
A = rand(1,1);
B = rand(1,1);
Z=1+sin((2*pi-A)*k*R);
Z1 = 1+sin((2*pi-B)*k*R1);
waterLevel = ((Z+Z1)/2)*(0.1+rand(1,1)*0.1); %0.07 for 1600-1899, 0.04 for 1900-2199, 0.07 for 4300-4499
% waterLevel = imfilter(randn(h, w), gKernel, 'circular');
waterLevel = waterLevel / max(waterLevel(:));
% waterLevelPrev = waterLevel;
% only for test.
waterLevel = zeros(h, w);

warp.Xs = zeros(h, w, nFrame);
warp.Ys = zeros(h, w, nFrame);
levels = zeros(h, w, nFrame);

% count=1;
% for freqrps=0.1:0.006:2*pi
%                     Z=sin((2*pi-freqrps)*k*R);

for i = 1:nFrame
    levels(:, :, i) = waterLevel-mean(waterLevel(:));
    
    warp.Xs(:, :, i) = alpha * imfilter(waterLevel, sobx, 'circular');
    warp.Ys(:, :, i) = alpha * imfilter(waterLevel, soby, 'circular');
     
%     lapimg = imfilter(waterLevel, lap, 'circular');    

%     waterLevelNext = 2 * waterLevel + c^2 * lapimg - waterLevelPrev;
%     
%     waterLevelPrev = waterLevel;
%     waterLevel = waterLevelNext;
      Z=1+sin((2*pi-i*0.2)*k*R);
      Z1 = 1+sin((2*pi-i*0.1)*k*R1);
      waterLevel = ((Z+Z1)/2)*(0.1+0.5*5);


end
warp.h = h;
warp.w = w;
warp.nFrame = nFrame;