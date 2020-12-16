%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
function [warp, levels] = getOceanWave(img, nFrame, alpha, c, WindDir, patchSize)

V = 200;%100;      % Wind speed m/s
A = 1;%1;        % Amplitude
g = 9.81;     % Gravitational constant
L = V*V/g;
WindDamp = 0.1;%0.01;       % Attenuation
WaveLimit = patchSize/100;

if size(img, 1) > 1 && size(img, 2) > 1
    h = size(img, 1);
    w = size(img, 2);
else
    w = img(1);
    h = img(2);
end

meshSize = w;   % FFT SIZE (128/256/384/512 for speed)

% GENERATE WAVE SPECTRUM
P = zeros(meshSize,meshSize);
k = zeros(meshSize,meshSize);
for m = 1:meshSize
    for n = 1:meshSize
        
        % CALCULATE WAVE VECTOR
        kx = 2*pi*(m-1-meshSize/2)/patchSize;
        ky = 2*pi*(n-1-meshSize/2)/patchSize;
        
        k2 = kx*kx + ky*ky;
        kxx = kx/sqrt(k2);
        kyy = ky/sqrt(k2);
        
        k(m,n) = sqrt(kx*kx + ky*ky);

        
        % WIND MODULATION
        w_dot_k = kxx*cos(WindDir) + kyy*sin(WindDir);
        
        % SPECTRUM AT GIVEN POINT
        P(m,n) = A*exp(- 1.0/((k(m,n)*L)^2))/(k(m,n)^4)*(w_dot_k^2);
        P(m,n) = P(m,n)*exp(-(k(m,n)^2) * WaveLimit^2);
        
        % FILTER WAVES MOVING IN THE WRONG DIRECTION
        if(w_dot_k<0)
            P(m,n) = P(m,n) * WindDamp;
        end
        
        if(kx == 0 && ky == 0)
            P(m,n) = 0;
        end
        
    end
end

%CALCULATE INITIAL SURFACE IN FREQUENCY DOMAIN
%RANDN - GAUSSIAN | RAND - NORMAL
H0 = 1./sqrt(2)*complex(randn(meshSize),randn(meshSize)).*sqrt(P);

 % GET MIRRORED VALUE OF INITIAL SURFACE
Hm = zeros(meshSize,meshSize);
for m = 1:meshSize
     for n=1:meshSize
          Hm(m,n) = conj(H0((meshSize-m+1),meshSize-n+1));
      end
end
 
%DISPERSION
W = sqrt(g.*k);

levels = zeros(h, w, nFrame);
warp.Xs = zeros(h, w, nFrame);
warp.Ys = zeros(h, w, nFrame);
tic;
for i = 1:nFrame
    time = c*i;
%     disp([wave_type ' Time Frame: ' num2str(i)])
    % Update according to the disperion relation
    Hkt = H0.*exp(1i.*W*time) + Hm.*exp(-1i.*W*time);
    
    % Generate HeightField at time t using ifft
    Ht = real(ifft2(Hkt));
    for m = 1:meshSize
        for n = 1:meshSize
             signCorrection = mod(m+n-2,2);
             if(signCorrection)
                 Ht(m,n) = -1.0*Ht(m,n);
             end
            
        end
    end
   
    zh = flip(flip(Ht,3),8); 
%     [Gx,Gy] = imgradientxy(zh,'central');
    [Gx,Gy] = imgradientxy(zh,'sobel');
%     pred_zh = cumsum([zh(1,:);Gx(2:end-1,:)],2);
%     pred_zh = cumsum(Gx,2)/10;
    
%     figure(2),subplot(1,2,1),imshow(mat2gray(zh));
%     figure(2),subplot(1,2,2),imshow(mat2gray(pred_zh));
%     cum_Gy = cumsum(Gy,2);
%     pred_zh = cum_Gx + cum_Gy;
    warp.Xs(:, :, i) = -alpha * Gx;
    warp.Ys(:, :, i) = -alpha * Gy;    
%     zh = (zh-min(zh(:)))./(max(zh(:))-min(zh(:)));
    levels(:, :, i) = zh;
end

    
