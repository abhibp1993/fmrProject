% Author: Abhishek N. Kulkarni
% Testing SDC

clear all, clc;

%% Create Car-SDC Object
c1 = SDC(1, 5, 5, Orientation.north, 1);

%% Test transition function
c1 = c1.step(2);    % Remember, step function returns same object type with modified object. 
c1.print();
