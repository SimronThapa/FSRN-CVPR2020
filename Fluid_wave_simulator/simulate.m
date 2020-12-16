%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
function imgCurr = simulate(img, warp_x,warp_y, isShown,nFrame)

[h, w, nChannel] = size(img);
% nFrame = size(warp.Xs, 3);

isShown = exist('isShown', 'var') && isShown;

[X, Y] = meshgrid(1:w, 1:h);

i = nFrame;
% for i = 1:nFrame

x_c = warp_x;
y_c = warp_y;
    % 
Xnew = reshape(X + x_c, h*w, 1);
Ynew = reshape(Y + y_c, h*w, 1);

valid = (Xnew >= 1 & Xnew <= w & Ynew >= 1 & Ynew <= h);

imgCurr = zeros(h, w, nChannel);
currFrame = zeros(h*w, 1);
for k = 1:nChannel
    currFrame(valid) = interp2(img(:, :, k), Xnew(valid), Ynew(valid));
    imgCurr(:, :, k) = reshape(currFrame, h, w);
end

% end;
