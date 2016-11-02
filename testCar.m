% Author: Abhishek Kulkarni
% Last Modified: 1 Nov 2016

clear all, clc

% Actions
STRAIGHT   = 'straight';
LEFT_TURN  = 'left';
RIGHT_TURN = 'right';

% Create Cars
c1 = Car(1, 5, 5, Orientation.north, 1);
fprintf('New Car:\t\t');
c1.print()

% Move Forward
c1 = action(c1, STRAIGHT);
fprintf('Straight Move:\t');
c1.print()

% Turn left
c1 = action(c1, LEFT_TURN);
fprintf('Turning Left:\t');
c1.print()

% Turn Right
c1 = action(c1, RIGHT_TURN);
fprintf('Turning Right:\t');
c1.print()

