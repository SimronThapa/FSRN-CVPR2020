%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
function [warp_map] = raytracing_im_generator_ST(im_rgb,this_depth,n1,n2,x,y)

step = 1;
[h,w,dim] = size(im_rgb);
% [Nx,Ny,Nz] = surfnorm(x,y,this_depth); 
% normal = cat(3, Nx, Ny, Nz); 

[Gx,Gy] = imgradientxy(this_depth);
normal_ori = ones(size(im_rgb));
normal_ori(:,:,1) = -Gx;
normal_ori(:,:,2) = -Gy;
normal = normal_ori./sqrt(Gx.^2 + Gy.^2 + 1);
% normal = permute(normal,[2 1 3]);
% nx = normal_test(:,:,1);
% ny = normal_test(:,:,2);
% nz = normal_test(:,:,3);

% figure(3),subplot(1,2,1),quiver3(x(1:10,1:10),y(1:10,1:10),this_depth(1:10,1:10),Nx(1:10,1:10),Ny(1:10,1:10),Nz(1:10,1:10));
% subplot(1,2,2),quiver3(x(1:10,1:10),y(1:10,1:10),this_depth(1:10,1:10),normal_test(1:10,1:10,1),normal_test(1:10,1:10,2),normal_test(1:10,1:10,3));
% % quiver3(x,y,this_depth,Nx,Ny,Nz) 
% % normal = permute(normal,[2 1 3]);
% figure(2),
% subplot(1,2,1),imshow(mat2gray(this_depth));
% subplot(1,2,2),imshow(mat2gray(normal_test));
% normal = imrotate(normal,90);
% out_im = zeros(size(normal));
% this_depth = 20;
s1 = zeros(size(normal));
s1(:,:,3) = -1;
s2 = refraction(normal,s1,n1,n2);
a = this_depth./s2(:,:,3);
x_c = round(a.*s2(:,:,1)/step,2);
y_c = round(a.*s2(:,:,2)/step,2);

warp_map = cat(3,x_c,y_c);



% y_i(y_i>256) = 256;
% x_j(x_j > 256) = 256;
% y_i(y_i<1) = 1;
% x_j(x_j<1) = 1;

% valid = (x_c >= 1 & x_c <= w & y_c >= 1 & y_c <= h);


% if dim == 3
%     for d=1:3
%         out_im(:,:,d) = interp2(x, y, im_rgb(:,:,d),x_j,y_i,'makima');
%     end
% else
%     out_im = interp2(X, Y, im_rgb,x_j,y_i,'makima');
% end

% out_im = zeros(h, w, dim);
% currFrame = zeros(h*w, 1);
% for k = 1:dim
%     currFrame(valid) = interp2(im_rgb(:, :, k), x_c(valid), y_c(valid));
%     out_im(:, :, k) = reshape(currFrame, h, w);
% end