clc; clear; close all; warning off all;

% memanggil menu "browse file"
[nama_file,nama_folder] = uigetfile('*.jpg');

% jika ada nama file yang dipilih maka akan mengekseskusi perintah dibawah
% ini

if ~isequal(nama_file,0)
    % membaca file citra rgb
    %membaca file citra rgb
    Img = im2double(imread(fullfile(nama_folder,nama_file)));
    % mengkonversi citra rgb menjadi citra greyscale
    Img_gray = rgb2gray(Img);
%     figure, imshow(Img)
%     figure, imshow(Img_gray)
    % mengkonversi citra grayscale menjadi citra biner
    bw = im2bw(Img_gray,.9);
%     figure, imshow(bw)
    % melakukan operasi komplemen
    bw = imcomplement(bw);
%     figure, imshow(bw)
    % melakukan operasi morfologi untuk menyempurnakan hasil segmentasi
    % 1. filling holes
    bw = imfill(bw,'holes');
%     figure, imshow(bw)
    % 2. area opening 
    bw = bwareaopen(bw,100);
%     figure, imshow(bw)
    %ekstraksi ciri warna rgb 
    R = Img(:,:,1);
    G = Img(:,:,2);
    B = Img(:,:,3);
    R(~bw) = 0;
    G(~bw) = 0;
    B(~bw) = 0;
%     RGB = cat(3,R,G,B);
%     figure, imshow(RGB)
    Red = sum(sum(R))/sum(sum(bw));
    Green = sum(sum(G))/sum(sum(bw));
    Blue = sum(sum(B))/sum(sum(bw));
    
    % menyusun variabel ciri_uji
    ciri_uji = [Red,Green,Blue];
    
    %memanggil model k-nn hasil pelatihan
    load Mdl

    %membaca kelas keluaran hasil pengujian
    hasil_uji = predict(Mdl,ciri_uji);
    
    %menampilkan citra asli dan kelas keluaran hasil pengujian
    figure, imshow(Img)
    title({['Nama File: ',nama_file],['Kelas Keluaran :',hasil_uji{1}]})
else
    % jika tidak ada nama file yang dipilih
    return
end