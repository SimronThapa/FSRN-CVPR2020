%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
function s2 = refraction(normal,s1,n1,n2)

this_normal = normal;
s1 = s1./sqrt((s1(:,:,1).^2) + (s1(:,:,2).^2) + (s1(:,:,3).^2));
term_1 = cross(this_normal,cross(-this_normal,s1,3),3);
term_2 = sqrt(1-(n1/n2)^2*sum(cross(this_normal,s1,3).*cross(this_normal,s1,3),3));
s2 = (n1/n2).*term_1 - this_normal.*term_2;
s2 = s2./sqrt((s2(:,:,1).^2) + (s2(:,:,2).^2) + (s2(:,:,3).^2));


