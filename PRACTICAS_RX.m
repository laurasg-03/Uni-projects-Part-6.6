%% PRÁCTICAS RX
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. INTRODUCCIÓN A MATLAB
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% • Standard formats image formats (eg: tif, .jpg, .png, .avi …) → imread
% • Medical image Formats (e.g., DICOM, NifTI, Analyze…) → dicomread, analyze75read …
% • Raw data formats from imaging systems (e.g., Raw data from a CT scanner) → Fopen, fread, fclose

%% Ver y cargar imagen en raw data
% Read File CT.raw
% Image Dimensions (pixels):
% 320 x 440 x 215
% 16 bits signed
% Little endian bit-order
clc; close all; clear all;

file = fopen('TC.raw', 'r', 'ieee-le'); % abrir el archivo en modo lectura y especificando que el orden de bits es little endian ('ieee-le'); para big-endian sería ('ieee-be')
image = fread(file, 320 * 440 * 215, 'int16'); % lee en formato imagen un número específico de elementos (320 * 440 * 215) del archivo y los almacena como datos de tipo int16 en el VECTOR imagen 
ima=reshape(image,320,440,215); %toma el vector image y lo remodela en una matriz tridimensional de dimensiones 320x440x215
fclose(file); % liberar los recursos asociados con el archivo

imshow(ima(:,:,116),[]); % muestra la imagen en el plano 116 de la pila de imágenes tridimensional; 
                         % [] valor mínimo se mapee a negro y el valor máximo se mapee a blanco
% imshow está diseñado para mostrar matrices 2D, y ima(23,:,:) o ima(:,23,:) es una matriz 3D. 



%% Visualizar los 3 planos
% Usar squeeze para reordenar la matriz para obtener una vista 2D

% • Plano Axial (XY):
% Visto desde arriba.
% El plano XY muestra cortes horizontales a lo largo del eje Z.
% Rotación alrededor del eje Z ([0,0,1]):
figure; imshow((ima(:,:,131)), []); %2D
imagen_rotada = imrotate(ima(:,:,131), 180); figure;imshow(imagen_rotada, []); %2D rotada 180º

figure; sliceViewer(ima); % 3D
img_rotate_XY=imrotate3(ima,90,[0,0,1]); figure; sliceViewer(img_rotate_XY) %3D rotada 90º

% • Plano Coronal (YZ):
% Visto desde el frente.
% El plano YZ muestra cortes verticales de adelante hacia atrás a lo largo del eje X.
% Este plano es útil para observar estructuras de adelante hacia atrás.
figure; imshow(squeeze(ima(210,:,:)), []); %2D
img_rotate_YZ=imrotate3(ima,90,[1,0,0]); figure; sliceViewer(img_rotate_YZ) %3D
%%
% • Plano Sagital (XZ):
% Visto desde un lado.
% El plano XZ muestra cortes verticales de lado a lado a lo largo del eje Y.
figure; imshow(squeeze(ima(:,369,:)), []); %2D
img_rotate_XZ=imrotate3(ima,90,[0,1,0]); figure; sliceViewer(img_rotate_XZ) %3D rotada 90º
sagital = permute(ima, [3, 1, 2]); figure;sliceViewer(sagital); %3D


orthosliceViewer(ima) %Fijarse en que las dimensaiones no estan proporcionales

%%
clc
cmap = parula(256);
s = sliceViewer(ima,'Colormap',cmap);

%% Histograma
clc
hist=histogram(ima(:),200);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. CT DATA PROCESSING
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We have tested the system by acquiring 180 projections of a 25-gr mouse spaced 1 degree from each other (0º-179º). Furthermore:
% X-Ray Tube settings: 40 Kvp, 0.7 mA
% Source to object distance: 60 mm
% Object to detector distance: 10 mm
% If the detector has an active area of 290 x 496 pixels (400 micron-pixels)
close all;
clear;
clc;

%% Q1 Determine the Field of View (FOV) of the imaging system
%The FOV of the system can be calculated by: 
% 1-determing the active area of the detector in mm
% 2-The FOV can be obtained multiplying the active are by the magnification factor.

pix_size=0.4; %mm (400 micron-pixels)
Active_Area_z_pix=496; % active area of 290 x 496 pixels
Active_Area_x_pix=290;

%Magnification
Source2Object=60;%mm % Source to object distance: 60 mm
Object2Detector=10;%mm % Object to detector distance: 10 mm
magnificacion = (Source2Object+Object2Detector)/Source2Object

%FOV of the projections is 
Active_Area_x_mm = (Active_Area_x_pix*pix_size)/magnificacion
Active_Area_z_mm=(Active_Area_z_pix*pix_size)/magnificacion 

%% Q2 Read the images into matlab
% After the data acquisition, the system provided the following files
clc;close all;clear all;

% • Raw_data_detector_final.raw:
% Este archivo contiene las 180 proyecciones adquiridas durante la exploración del ratón EN EL EJE Z!
% Cada píxel está codificado con 8 bits de profundidad, lo que significa que puede tener un valor entre 0 y 255, representando diferentes niveles de intensidad de la radiación absorbida.
% Las proyecciones se adquirieron moviendo la fuente de rayos X y/o el detector alrededor del objeto (en este caso, el ratón) en incrementos de 1 grado desde 0° a 179°, registrando así diferentes ángulos de visión del objeto.
% named Projections (290x496x180 elements of 1 byte each).
file1= fopen('Raw_data_detector_final.raw');
image1 = fread(file1, 'uint8');
Projections=reshape(image1,290,496,[]);
fclose(file1);

% fID=fopen("Raw_data_detector_final.raw","r");
% Projections=fread(fID,290*496*180,"uint8");
% fclose(fID);
% Projections=reshape(Projections,290,496,180);


% • flood_image_final.raw:
% Este archivo contiene una imagen adquirida con la misma configuración del tubo de rayos X y del detector que se utilizó para la adquisición de las proyecciones del ratón, sin embargo, en esta imagen no hay ningún objeto presente en el campo de visión del sistema.
% Esta imagen se adquiere con el fin de proporcionar una referencia para la calibración y corrección de la atenuación no uniforme en las imágenes proyectadas del objeto.
file2 = fopen('flood_image_finale.raw');
image2 = fread(file2, 'uint8');
flood=reshape(image2,290,496,[]);
fclose(file2);

% fID=fopen("flood_image_final.raw","r");
% flood=fread(fID,290*496,"uint8");
% fclose(fID);
% flood=reshape(flood,290,496);


% • dark_current_final.raw:
% Este archivo contiene una imagen adquirida con la misma configuración del detector que se utilizó para la adquisición de las proyecciones del ratón. Sin embargo, en esta imagen el tubo de rayos X se encuentra apagado, por lo que no hay radiación proveniente de la fuente.
% Esta imagen se utiliza para capturar el "ruido" o la señal de fondo inherente al detector cuando no hay radiación incidente, como la corriente oscura y el ruido electrónico. Se utiliza para corregir este ruido en las imágenes proyectadas del objeto.
file3 = fopen('dark_current_final.raw');
image3 = fread(file3, 'uint8');
dark=reshape(image3,290,496,[]);
fclose(file3);

% fID=fopen("dark_current_final.raw","r");
% dark=fread(fID,290*496,"uint8");
% fclose(fID);
% dark=reshape(dark,290,496);


subplot(1,3,1);
imshow(imrotate(Projections(:,:,1),-90), []); % el 1 son los grados de rotación y el -90 es porque la imagen de a 0º está tumbada hacia la derecha 
subplot(1,3,2);
imshow(imrotate(flood,-90),[]);
subplot(1,3,3);
imshow(imrotate(dark,-90),[]);

%% Q3: Determine the position of the defective lines in the detector according to the manufacturer of the flat panel 
% detector the projections acquired with this system contains defective lines (complete lines stuck at 0 or at 255) in both 
% directions that we need to correct before any further processing. In order to do so:
z_project=sum(Projections,3)/size(Projections,3);
subplot(1,1,1);
imshow(z_project,[]);

%%
clc
stuck1_x_p= find(Projections(:,1) == 255); %Lines with value 255 along the X dimension 
stuck0_x_p= find(Projections(:,1) == 0); %Lines with value 0 along the X dimension
stuck1_z_p= find(Projections(1,:) == 255); %Lines with value 255 along the Z dimension
stuck0_z_p= find(Projections(1,:) == 0); %Lines with value 0 along the Z dimension

stuck1_x_f= find(flood(:,1) == 255); %Lines with value 255 along the X dimension 
stuck0_x_f= find(flood(:,1) == 0); %Lines with value 0 along the X dimension
stuck1_z_f= find(flood(1,:) == 255); %Lines with value 255 along the Z dimension
stuck0_z_f= find(flood(1,:) == 0); %Lines with value 0 along the Z dimension

stuck1_x_d= find(dark(:,1) == 255); %Lines with value 255 along the X dimension 
stuck0_x_d= find(dark(:,1) == 0); %Lines with value 0 along the X dimension
stuck1_z_d= find(dark(1,:) == 255); %Lines with value 255 along the Z dimension
stuck0_z_d= find(dark(1,:) == 0); %Lines with value 0 along the Z dimension


%Example: If a defective line is in position X=ii, the correction will be
%X[ii]= X[ii-1]+X[ii+1] /2.0




%TIP: Use the indexes obtained in Q3

%% Q4: Correct defective lines in the 3 images 
% flood
flood(stuck0_x_f,:)=[flood(stuck0_x_f-1,:)+flood(stuck0_x_f+1,:)] /2.0;
flood(stuck1_x_f,:)=[flood(stuck1_x_f-1,:)+flood(stuck1_x_f+1,:)] /2.0;
flood(:,stuck0_z_f)=[flood(:,stuck0_z_f-1)+flood(:,stuck0_z_f+1)] /2.0;
flood(:,stuck1_z_f)=[flood(:,stuck1_z_f-1)+flood(:,stuck1_z_f+1)] /2.0;

imshow(imrotate(flood,-90),[]);

% dark
dark(stuck0_x_f,:)=[dark(stuck0_x_f-1,:)+dark(stuck0_x_f+1,:)] /2.0;
dark(stuck1_x_f,:)=[dark(stuck1_x_f-1,:)+dark(stuck1_x_f+1,:)] /2.0;
dark(:,stuck0_z_f)=[dark(:,stuck0_z_f-1)+dark(:,stuck0_z_f+1)] /2.0;
dark(:,stuck1_z_f)=[dark(:,stuck1_z_f-1)+dark(:,stuck1_z_f+1)] /2.0;

imshow(imrotate(dark,-90),[]);


% Projections
clc
Projections(stuck0_x_f,:,:)=(Projections(stuck0_x_f-1,:,:)+Projections(stuck0_x_f+1,:,:)) /2.0;
Projections(stuck1_x_f,:,:)=(Projections(stuck1_x_f-1,:,:)+Projections(stuck1_x_f+1,:,:)) /2.0;
Projections(:,stuck0_z_f,:)=(Projections(:,stuck0_z_f-1,:)+Projections(:,stuck0_z_f+1,:)) /2.0;
Projections(:,stuck1_z_f,:)=(Projections(:,stuck1_z_f-1,:)+Projections(:,stuck1_z_f+1,:)) /2.0;


flood=medfilt2(flood);
dark=medfilt2(dark);

subplot(1,3,1);
imshow(imrotate(Projections(:,:,1),-90), []);
subplot(1,3,2);
imshow(imrotate(flood,-90),[]);
subplot(1,3,3);
imshow(imrotate(dark,-90),[]);

%% Q5: Correct dark current in all the images (flood and projection)
clc
% correct the flood image
% store result in the flood variable

flood=  flood-dark; % flood image with corrected lines and dark_current
imshow(imrotate(flood(:,:,1),-90), []);

%Ensure that there is no numbers < 0 in flood  after this correction
a=find(flood(:,:)<0);

% Correct the projections

Projections=Projections-dark; % Projections with corrected lines and dark_current
Projections(Projections<0)=0;
b=find(Projections(:,:)<0)
%Ensure that there is no numbers < 0 after this correction


% plot one of the resulting projections after this correction

    %Uncomment when ready
subplot(1,1,1)
imshow(imrotate(Projections(:,:,1),-90), []); %Corrected projections, no defective lines, No dark component


%% Q6: Obtain attenuation projections and store them to disk
%I0 is the Flood image (Think about it...)
%I is the image provided by the detector (projections)
%Consequently p(ri)=-Ln(I/Io)

%1. As in a previous step, we create a flood 3d matrix using repmat
flood3D=repmat(flood,1,1,180);

%2. p(ri)=-Ln(Projections./flood3D)
Projections_attenuation=-log(Projections./flood3D);

%3. Save the file, so you can open it with imageJ
fID=fopen("attenuation_projections.raw","w");
fwrite(fID,Projections_attenuation,'single');
fclose(fID);

%4. See the processed data using implay. Try adjusting the pixel range (tools--> Colormap --> Specify range 
% Try for example -0.5 to 3 
implay(imrotate3(Projections_attenuation,-90,[0,0,1]));


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. ANALYTIC IMAGE RECONSTRUCTION
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Q1: Load the projections dataset and show the projection data and its radon transform (sinogram):
% Load the file "attenuation_projections_high_SNRf.raw" and show the sinogram of the central slice as well as the first projection image
% NOTE: Projections are the same we generated in session 2, adapted for reconstruction using paralell rays
% size : 290 x 496 x 180 pixels (Float 32 bits) 
%clear the workspace in each new execution

close all;
clear all;
clc;

%Load the  projections
fID=fopen("attenuation_projections_high_SNRf.raw","r");
Projections=fread(fID,290*496*180,"float");
Projections=reshape(Projections,290,496,180);
fclose(fID);

%draw projection at 90 deg and the central slice sinogram 
subplot(1,2,1);
imshow(imrotate(Projections(:,:,1),-90), []); 
title('Projection at 90 deg');

subplot(1,2,2);
imshow(imrotate(squeeze(Projections(:,248,:)),-90),[]); % Corte central;  squeeze elimina las dimensiones singleton (dimensiones de tamaño 1) para que el resultado sea una matriz 2D.
% Al seleccionar Projections(:, 248, :), estamos extrayendo todas las proyecciones de la columna central (la rebanada central) a través de todos los ángulos disponibles (180 en este caso).
% Projections(:, 248, :) es una matriz de dimensiones 290 x 180, donde:
% • 290 son las posiciones en el eje x (a lo largo de la rebanada central en el eje y).
% • 180 son los diferentes ángulos de proyección.
title('Radon transform central slice '); % La proyección de la columna vertebral de la rata rotada 180º


%% Q2 Reconstruct the central slice using the backprojection and Filtered BackProjetion algorithms: 
% Reconstruct the central slice of the volume  with the BP and the FBP algorithms and draw the results.
% Assume the projections were acquired using a Generation 1 scanner (Translation and Rotation)
%Reconstruct the central slice of the sinogram using 

% a) Backprojection algorithm (BP)
slice_unfilt=iradon(squeeze(Projections(:,248,:)),0:179,'linear','none'); % Retroproyección sin filtrar

% b) Filtered BackProjection with a Ramp Filter (FBP)
slice_filt=iradon(squeeze(Projections(:,248,:)),0:179,'linear','Ram-Lak');% Filtro rampa: Máxima nitidez en bordes y detalles finos, con mayor ruido.

% slice_shepp = iradon(sinogram, 0:179, 'linear', 'Shepp-Logan'); %  Buen equilibrio entre nitidez y reducción de ruido.
% slice_cosine = iradon(sinogram, 0:179, 'linear', 'Cosine'); % Suficiente suavizado para reducir el ruido sin perder muchos detalles.
% slice_hamming = iradon(sinogram, 0:179, 'linear', 'Hamming'); % Suaviza la imagen y reduce los artefactos de ondulación, buen para imágenes más limpias.
% slice_hann = iradon(sinogram, 0:179, 'linear', 'Hann'); % Similar a Hamming, con un buen balance entre reducción de ruido y suavizado.

%draw the reconstructed slices (BP and FBP)
subplot(1,2,1);
imshow(slice_unfilt, []);
title('Unfiltered Backprojection');

subplot(1,2,2);
imshow(slice_filt,[]);
title('Filtered Backprojection');


%% Q3: Reconstruct the complete volume using the backprojection  and FBP algorithms
% Reconstruct all slices using the BP and FBP algorithms [store the data into disk so you can later open it with imageJ]. 
% Show  the sagital, axial and tangential slices of the reconstructed volume (central slices) for the BP and FBP algorithms. 
% Determine the size of the reconstructed volumes (BP and FBP) and create variables to store them 

% Inicialización y Definición de Dimensiones
dimx = size(slice_filt, 1); % Dimensiones de la rebanada reconstruida previamente
dimy = size(slice_filt, 2);
num_Slices = 496; % Número de rebanadas en el volumen
Volume_unfilt = zeros(dimx, dimy, num_Slices); % Inicialización del volumen sin filtrar
Volume_filt = zeros(dimx, dimy, num_Slices); % Inicialización del volumen filtrado

%Reconstruct all slices 
for i=1:num_Slices
    Volume_unfilt(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','none');
    Volume_filt(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','Ram-Lak');
end

%Save data to disk
fID=fopen("BackProjection_Reconstructed.raw","w");
fwrite(fID,Volume_unfilt,'single');
fclose(fID);

fID=fopen("FBackProjection_Reconstructed.raw","w");
fwrite(fID,Volume_filt,'single');
fclose(fID);

% Show central coronal views of the reconstructed volumes
figure
subplot(1,2,1);
imshow(imrotate(squeeze(Volume_unfilt(102,:,:)),-90), []);
title('Coronal View BP');
subplot(1,2,2);
imshow(imrotate(squeeze(Volume_filt(102,:,:)),-90), []);
title('Coronal View FBP');

% Coronal View (y-z): Se obtiene de la posición central en el eje x (Volume_unfilt(round(dimx/2), :, :)).
% Axial View (x-y): Se obtiene de la posición central en el eje z (Volume_unfilt(:, :, round(num_Slices/2))).
% Sagittal View (x-z): Se obtiene de la posición central en el eje y (Volume_unfilt(:, round(dimy/2), :)).


%Plot Sagital views
figure
subplot(1,2,1);
imshow(imrotate(squeeze(Volume_unfilt(:,102,:)),-90), []);
title('Sagital View BP');
subplot(1,2,2);
imshow(imrotate(squeeze(Volume_filt(:,102,:)),-90), []);
title('Sagital View FBP');


%PLot axial slices
figure
subplot(1,2,1);
imshow(Volume_unfilt(:,:,248), []);
title('Axial View BP');
subplot(1,2,2);
imshow(Volume_filt(:,:,248), []);
title('Axial View FBP');


%Alternatively, you can simply use the orthoSliceViewer
figure; orthosliceViewer(Volume_filt);
figure; orthosliceViewer(Volume_unfilt);

%% Q4: Improve the quality of the FBP reconstructed images
% FBP reconstructed images appear to be very noisy...  why?
% Try different strategies to improve SNR of the reconstructed images (Different reconstructionfilters / cutoff freqencies / interpolation), store data to disk and chek the results with AMIDE and or ImageJ

%Declare variables to store the images reconstructed with different
%parameters

Volume_filt2=zeros(dimx,dimy,num_Slices);
Volume_filt3=zeros(dimx,dimy,num_Slices);
Volume_filt4=zeros(dimx,dimy,num_Slices);

%Reconstruct the volumes
for i=1:num_Slices
    Volume_filt(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','Hann',0.25);
    Volume_filt2(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','Shepp-Logan',0.3);
    Volume_filt3(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','Cosine',0.15);
    Volume_filt4(:,:,i)=iradon(squeeze(Projections(:,i,:)),0:179,'linear','Hamming',0.7);   
end


%Save data to disk
% fID=fopen("FBP_Reconstructed_Hann.raw","w");
% fwrite(fID,Volume_filt,'single');
% fclose(fID);

% %Show reconstructed views of the results
figure
subplot(1,2,1);
imshow(imrotate(squeeze(Volume_filt(102,:,:)),-90), []);
title('Coronal View Hann Filter');

subplot(1,2,2);
imshow(imrotate(squeeze(Volume_filt2(102,:,:)),-90), []);
title('Coronal View Sheep-Logan Filter');

%% Q5: show the results using a volume rendering
% show the results using a volume rendering (for example, Maximun Intensity Projection, MIP)
% Reference: IEEE Trans Med Imaging1989;8(4):297-30. doi: 10.1109/42.41482. Three-dimensional display in nuclear medicine
% NOTE: Check also imagesc function
%Render the volume to show the coronal view
%The render algorithm, MIPs simply find the maximun along the viewing
%direction. We are doing it only for 2 angles, if made for more, is
%possible to create a movie

figure;
RenderImage=rot90(squeeze(max(Volume_filt,[],1)),3);
imshow(RenderImage,[]);
% Alternatively, you can do it with function imagesc
figure; imagesc(rot90(squeeze(max(Volume_filt,[],1)),3)); colormap('bone');

%Render the volume to show the sagital view
figure;
RenderImage=rot90(squeeze(max(Volume_filt,[],2)),3);
imshow(RenderImage,[]);

%Alternatively, you can do it with function imagesc
figure; imagesc(rot90(squeeze(max(Volume_filt,[],2)),3)); colormap('bone');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. ITERATIVE IMAGE RECONSTRUCTION
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T'Phe file 'centralSliceNoiseFree.raw'   (256*256 pixels,  float 32 bit) contains an almost noise-free image, from a X-ray Computed tomography scanner (single slice). This image is intended to serve as a groundtruth to evaluate the performance of the different reconstruction approaches.
% Unfortunately, the data we acquired with our scaner  (parallel projections data), have a certain level of noise that will be propagated to the reconstructed slice if we do not deal with it in a proper way. These data can be found in the file
%  rojectionsCentralSlice.raw'  (367*180 pixels,  float 32 bit)
clc; clear all; close all;

% Carga del Sinograma Ruidoso:
file2 = fopen('ProjectionsCentralSlice.raw'); % Sinograma
image2 = fread(file2, 367*180, 'float32');
imagen_ruido2=reshape(image2,367,180, []);
fclose(file2)

figure;
imshow(imrotate(imagen_ruido2, -90), []);
title('Noisy Sinogram');


%% Q1. Load the ideal image (No noise), store it in centralSlice variable
file1 = fopen('centralSliceNoiseFree.raw');
image = fread(file1, 256*256, 'float32');
centralSlice=reshape(image,256,256, []);
fclose(file1)
figure;
imshow(imrotate(centralSlice, -90), []);
title('Noise-Free Image');


% Reconstrucción con Filtro de Rampa (Ramp Filter):
centralSliceNoise = iradon(imagen_ruido2, 0:179, 'linear', 'Ram-Lak');

% Reconstrucción con Filtro Shepp-Logan:
centralSliceNoiseShepL = iradon(imagen_ruido2, 0:179, 'linear', 'Shepp-Logan');

figure;
subplot(1, 3, 1);
imshow(imrotate(centralSlice, -90), []);
title('Noise-Free Image');
colormap('bone');

subplot(1, 3, 2);
imshow(imrotate(centralSliceNoise, -90), []);
title('Reconstructed with Ramp Filter');
colormap('bone');

subplot(1, 3, 3);
imshow(imrotate(centralSliceNoiseShepL, -90), []);
title('Reconstructed with Shepp-Logan Filter');
colormap('bone');

%% Q2: Reconstruct the data using the ART algorithm
clc
% ART= initial image estimate, will be updated at each iteration of the algorithm
% In the initial iteration, ART has all its pixels equal to Zero
% 
% For each iteration
% PROY_ART = projections for the ART image estimate of the current iteration
% CORR_ART = Correction to apply in the current iteration. This is obtained  by comparing the Projections of the image estimate and the acquired projections
% NUM_ART= Image obtained by retroproyecting  the previous result.This image, will be used to update the ART image in the current iteration, will be the numerator of the correction
% ART=  ART + (NUM_ART./DENOM_ART); The image is updated  in each iteration using this equation  
% NOTE: DENOM_ART is a precalculated matrix of weighting weighting factors among all pixels    

%YOUR ART IMPLEMENTATION HERE

figure                       %FIGURE IN WHICH WE WILL BE SHOWING THE RESULT AFTER EACH ITERATION
niter=120;                   %NUMBER OF ITERATIONS
%-------- ART --------------------------------

ART = zeros(256,256);                                   %INITIAL IMAGE (ART)

%Estimate the art denominator, look at the information available to you... 

DENOM_ART = ones(256,256);  %ART denominator 
DENOM_ART = 256;



for iter=1:niter
    %--------- ART --------------------------------
PROY_ART = radon(ART,0:179);  % 1-projections for the ART image estimate of the current iteration
CORR_ART = imagen_ruido2 - PROY_ART;  % 2-Correction to apply in the current iteration. This is obtained  by comparing the Projections of the image estimate and the acquired projections
NUM_ART =  iradon(CORR_ART,0:179,"none",256);  % 3-Image obtained by retroproyecting  the previous result.This image, will be used to update the ART image in the current iteration, will be the numerator of the correction
ART = ART + (NUM_ART./DENOM_ART);    %4 The image is updated  in each iteration using this equation   ART
    
    %Show the results of each iteration
    stringp=sprintf("ART iteration # %d", iter);
    imagesc(imrotate(ART,-90)),xlabel( stringp),colormap("gray")


drawnow; 
pause(.1)

end










