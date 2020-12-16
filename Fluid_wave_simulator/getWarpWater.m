function [warp, levels] = getWarpWater(img, nFrame, c, alpha)
% Created by Yuandong Tian, Oct. 5, 2009
% Email: yuandong@andrew.cmu.edu
% a simple simulator water simulator with random initial conditions. Note the
% first frame is no distorted. 

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
A = randn(h, w);
waterLevel = imfilter(A, gKernel, 'circular');
waterLevel = waterLevel / max(waterLevel(:));
waterLevelPrev = waterLevel;
% only for test.
waterLevel = zeros(h, w);

warp.Xs = zeros(h, w, nFrame);
warp.Ys = zeros(h, w, nFrame);
levels = zeros(h, w, nFrame);

for i = 1:nFrame
    levels(:, :, i) = waterLevel-mean(waterLevel(:));
    
    warp.Xs(:, :, i) = alpha * imfilter(waterLevel, sobx, 'circular');
    warp.Ys(:, :, i) = alpha * imfilter(waterLevel, soby, 'circular');
     
    lapimg = imfilter(waterLevel, lap, 'circular');
    
%      [warp.Xs(:, :, i), warp.Ys(:, :, i)] = gradient(waterLevel);
%      warp.Xs(:, :, i) = warp.Xs(:, :, i) * alpha;
%      warp.Ys(:, :, i) = warp.Ys(:, :, i) * alpha;

    waterLevelNext = 2 * waterLevel + c^2 * lapimg - waterLevelPrev;
    
    waterLevelPrev = waterLevel;
    waterLevel = waterLevelNext;
end
warp.h = h;
warp.w = w;
warp.nFrame = nFrame;